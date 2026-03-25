import os
from pathlib import Path

import joblib
import mt_prior
import numpy as np
from obspy import geodetics

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

# --- 1. Model Definition (Must match training exactly) ---
DEPTH = 4
WIDTH = 256
DROPOUT = 0.0
DEFAULT_MODEL_FILE = "checkpoint_N1000000_seed38_epoch999.pt"
DEFAULT_X_SCALER_FILE = "x_scaler_38_1000000.pkl"
DEFAULT_Y_SCALER_FILE = "y_scaler_38_1000000.pkl"
REFERENCE_MW = 5.0


if TORCH_AVAILABLE:
    class MLP(nn.Module):
        def __init__(self, n_in, depth=DEPTH, width=WIDTH, dropout=DROPOUT):
            super().__init__()
            layers = [nn.Linear(n_in, width), nn.ReLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(width, width), nn.ReLU()]
            self.backbone = nn.Sequential(*layers)
            self.dropout = nn.Dropout(dropout)
            self.out = nn.Linear(width, 1)

        def forward(self, x):
            x = self.backbone(x)
            x = self.dropout(x)
            return self.out(x)
else:
    class MLP:  # pragma: no cover - fallback type for import-time compatibility
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "torch is not installed; ML power inference is unavailable in this environment."
            )


# --- 2. Helper: Mag -> Moment Tensor ---
def scalar_moment_from_mw(mag):
    """
    Hanks & Kanamori (1979): M0 = 10^(1.5*Mw + 9.1) for N-m.
    """
    return 10 ** (1.5 * mag + 9.1)


def magnitude_to_moment_tensor_isotropic(mag):
    """
    Converts Moment Magnitude (Mw) to a 6-component Moment Tensor
    assuming a pure Isotropic (Explosion) source.

    Hanks & Kanamori (1979): M0 = 10^(1.5*Mw + 9.1) (for N-m)
    """
    # Calculate Scalar Moment M0
    # Note: Ensure your MLP was trained on N-m. If dyne-cm, change 9.1 to 16.1.
    m0 = scalar_moment_from_mw(mag)

    # For isotropic, diagonal terms are M0, off-diagonals are 0.
    # [m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]
    return np.array([m0, m0, m0, 0.0, 0.0, 0.0])


def log_power_magnitude_shift(mag, mw_ref=REFERENCE_MW):
    """
    Shift log integrated power from a reference Mw to a target Mw.

    Power scales with the square of the waveform amplitude, while waveform
    amplitude scales linearly with seismic moment M0. In natural-log units:

        log P(Mw) = log P(Mw_ref) + 2 * log(M0(Mw) / M0(Mw_ref))

    which simplifies to:

        log P(Mw) = log P(Mw_ref) + 3 * ln(10) * (Mw - Mw_ref)
    """
    return 3.0 * np.log(10.0) * (float(mag) - float(mw_ref))


def resolve_repo_root():
    env_root = os.environ.get("SEISMIC_OED_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parent


def resolve_power_model_paths(model_path=None, x_path=None, y_path=None):
    root = resolve_repo_root()
    model = Path(model_path) if model_path is not None else root / DEFAULT_MODEL_FILE
    x_scaler = Path(x_path) if x_path is not None else root / DEFAULT_X_SCALER_FILE
    y_scaler = Path(y_path) if y_path is not None else root / DEFAULT_Y_SCALER_FILE
    return str(model), str(x_scaler), str(y_scaler)


def map_sensor_fidelity_to_gaussian_variance(
    sensor_fidelity, strategy="clamped_linear", raw_min=0.0, raw_max=0.2, fixed_value=2.0
):
    fidelity = np.asarray(sensor_fidelity, dtype=float)
    if strategy == "direct":
        return fidelity.copy()
    if strategy == "fixed_nominal":
        return np.full_like(fidelity, float(fixed_value), dtype=float)
    if strategy == "clamped_linear":
        denom = max(float(raw_max) - float(raw_min), 1e-12)
        scaled = 1.0 + 2.0 * ((fidelity - float(raw_min)) / denom)
        return np.clip(scaled, 1.0, 3.0)
    raise ValueError(f"Unknown fidelity mapping strategy: {strategy}")


def training_mt_norm_reference(mw_ref=5.0):
    mt = magnitude_to_moment_tensor_isotropic(mw_ref)
    return float(np.sqrt(0.5 * np.sum(mt**2)))


def evaluate_power_domain_gates(
    theta,
    sensors,
    gaussian_variance,
    latlon_abs_bound=2.0,
    depth_min_m=5000.0,
    depth_max_m=20000.0,
    gauss_min=1.0,
    gauss_max=3.0,
    mt_norm_ratio_min=0.5,
    mt_norm_ratio_max=2.0,
):
    src_lat, src_lon, src_depth_km, src_mag = theta
    sens_lat = sensors[:, 0]
    sens_lon = sensors[:, 1]
    gaussian_variance = np.asarray(gaussian_variance, dtype=float)

    local_lat = sens_lat - src_lat
    local_lon = sens_lon - src_lon
    depth_m = float(src_depth_km) * 1000.0

    lat_gate = np.abs(local_lat) <= float(latlon_abs_bound)
    lon_gate = np.abs(local_lon) <= float(latlon_abs_bound)
    depth_gate = (depth_m >= float(depth_min_m)) & (depth_m <= float(depth_max_m))
    gauss_gate = (gaussian_variance >= float(gauss_min)) & (
        gaussian_variance <= float(gauss_max)
    )

    # The NN now always sees a fixed-norm Mw=5 isotropic tensor and magnitude
    # differences are handled by an output-space log-power shift. That keeps the
    # MT input features pinned to the training norm.
    ratio = 1.0
    mt_gate_scalar = True
    mt_gate = np.full(len(sensors), mt_gate_scalar, dtype=bool)

    enabled = lat_gate & lon_gate & depth_gate & gauss_gate & mt_gate
    gate_masks = {
        "lat_local": lat_gate,
        "lon_local": lon_gate,
        "depth_m": np.full(len(sensors), depth_gate, dtype=bool),
        "gaussian_variance": gauss_gate,
        "mt_norm": mt_gate,
    }

    return enabled, gate_masks, {
        "mt_norm_ratio": ratio,
        "depth_m": depth_m,
    }


# --- 3. The Interface Class ---
class SeismicPowerInterface:
    def __init__(self, model_path, x_scaler_path, y_scaler_path):
        self.device = (
            "cpu"  # CPU is usually faster for scalar/small-batch inference loops
        )

        # Load Model
        self.model = MLP(n_in=11, depth=DEPTH, width=WIDTH, dropout=DROPOUT).to(
            self.device
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device(self.device))
        )
        self.model.eval()  # CRITICAL: Set to eval mode

        # Load Scalers
        self.x_scaler = joblib.load(x_scaler_path)
        self.y_scaler = joblib.load(y_scaler_path)

    def predict_log_power(self, theta, sensors, gaussian_variance=None, mt_override=None):
        """
        Predicts log integrated power for a single event `theta` against multiple `sensors`.

        Inputs
        ------
        theta : np.array (4,)
            [Lat, Lon, Depth, Mag]
        sensors : np.array (N, 5)
            [Lat, Lon, Fidelity/Variance, ..., Type]

        Returns
        -------
        log_power_pred : np.array (N,)
            Predicted log power for each sensor.

        Notes
        -----
        `mt_override`, when provided, is interpreted as a normalized 6-component
        MT direction in Voigt order. It is rescaled to the training reference
        norm before entering the NN feature vector.
        """
        # Unpack event
        src_lat, src_lon, src_depth, src_mag = theta

        # Unpack sensors
        # Assuming sensor columns: [Lat, Lon, Fidelity, ..., Type]
        sens_lat = sensors[:, 0]
        sens_lon = sensors[:, 1]
        sens_fidelity = sensors[:, 2] if gaussian_variance is None else gaussian_variance

        # 1. Compute Distances
        # Use obspy's vector-capable function if possible, or loop.
        # For safety/clarity here, we use the scalar function in a list comp
        # (but vectorizing this calculation in numpy is better if N is large).
        dists_deg = []
        for sl, slon in zip(sens_lat, sens_lon):
            dists_deg.append(geodetics.locations2degrees(src_lat, src_lon, sl, slon))
        dists_km = geodetics.degrees2kilometers(np.array(dists_deg))

        # 2. Get reference-norm moment tensor.
        # The NN was trained with fixed Mw=5 tensors, so keep the MT input at
        # the training norm and scale the predicted log-power afterward.
        if mt_override is None:
            mt_direction = mt_prior.isotropic_mt_unit()
        else:
            mt_direction = mt_prior.normalize_mt(np.asarray(mt_override, dtype=float))
            if mt_direction.shape != (6,):
                raise ValueError("mt_override must be a single normalized 6-vector.")
        mt = mt_direction * training_mt_norm_reference(REFERENCE_MW)

        # 3. Build Input Matrix (N_sensors x 11 features)
        # Columns: [Lat, Lon, Depth, Dist_km, Variance, m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]
        # We need to tile the source-specific features to match N sensors
        N = len(sensors)

        # Create the feature matrix
        features = np.zeros((N, 11))

        # Fill sensor-specific columns (source-relative lat/lon in degrees)
        features[:, 0] = sens_lat - src_lat
        features[:, 1] = sens_lon - src_lon

        # Fill Source-specific columns (repeated for all sensors)
        # Model expects source depth in meters
        features[:, 2] = src_depth * 1000.0
        features[:, 3] = dists_km
        features[:, 4] = sens_fidelity

        # Fill Moment Tensor (repeated)
        features[:, 5:11] = np.tile(mt, (N, 1))

        # 4. Scale Inputs
        features_scaled = self.x_scaler.transform(features).astype(np.float32)
        abs_z = np.abs(features_scaled)
        if np.any(abs_z > 10.0) and not hasattr(self, "_printed_ood_warning"):
            max_flat_idx = int(np.argmax(abs_z))
            max_row, max_col = np.unravel_index(max_flat_idx, abs_z.shape)
            max_abs_z = float(abs_z[max_row, max_col])
            print(
                f"[ML power OOD warning] abs(z)>10 detected; max_abs_z={max_abs_z:.3f} at feature_index={max_col}"
            )
            self._printed_ood_warning = True

        # 5. Inference
        with torch.no_grad():
            tensor_x = torch.from_numpy(features_scaled).to(self.device)
            pred_std = self.model(tensor_x).view(-1).cpu().numpy()

        # 6. Inverse Transform Outputs
        # y_scaler expects (N, 1)
        log_power_pred = self.y_scaler.inverse_transform(
            pred_std.reshape(-1, 1)
        ).ravel()

        # Adjust the physical log-power for the requested source magnitude
        # without pushing the MT input features out of the training domain.
        log_power_pred = log_power_pred + log_power_magnitude_shift(
            src_mag, mw_ref=REFERENCE_MW
        )

        return log_power_pred


# Global instance variable (to be initialized by main code)
_POWER_MODEL = None


def get_power_model(model_path=None, x_path=None, y_path=None):
    """Singleton accessor to avoid reloading weights."""
    global _POWER_MODEL
    if _POWER_MODEL is None:
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is not installed; cannot initialize seismic power model."
            )
        model_path, x_path, y_path = resolve_power_model_paths(model_path, x_path, y_path)
        _POWER_MODEL = SeismicPowerInterface(model_path, x_path, y_path)
    return _POWER_MODEL
