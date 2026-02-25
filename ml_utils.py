import joblib
import numpy as np
import torch
import torch.nn as nn
from obspy import geodetics

# --- 1. Model Definition (Must match training exactly) ---
DEPTH = 4
WIDTH = 256
DROPOUT = 0.0


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


# --- 2. Helper: Mag -> Moment Tensor ---
def magnitude_to_moment_tensor_isotropic(mag):
    """
    Converts Moment Magnitude (Mw) to a 6-component Moment Tensor
    assuming a pure Isotropic (Explosion) source.

    Hanks & Kanamori (1979): M0 = 10^(1.5*Mw + 9.1) (for N-m)
    """
    # Calculate Scalar Moment M0
    # Note: Ensure your MLP was trained on N-m. If dyne-cm, change 9.1 to 16.1.
    m0 = 10 ** (1.5 * mag + 9.1)

    # For isotropic, diagonal terms are M0, off-diagonals are 0.
    # [m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]
    return np.array([m0, m0, m0, 0.0, 0.0, 0.0])


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

    def predict_log_power(self, theta, sensors):
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
        """
        # Unpack event
        src_lat, src_lon, src_depth, src_mag = theta

        # Unpack sensors
        # Assuming sensor columns: [Lat, Lon, Fidelity, ..., Type]
        sens_lat = sensors[:, 0]
        sens_lon = sensors[:, 1]
        sens_fidelity = sensors[:, 2]  # This maps to 'Gaussian_variance'

        # 1. Compute Distances
        # Use obspy's vector-capable function if possible, or loop.
        # For safety/clarity here, we use the scalar function in a list comp
        # (but vectorizing this calculation in numpy is better if N is large).
        dists_deg = []
        for sl, slon in zip(sens_lat, sens_lon):
            dists_deg.append(geodetics.locations2degrees(src_lat, src_lon, sl, slon))
        dists_km = geodetics.degrees2kilometers(np.array(dists_deg))

        # 2. Get Moment Tensor (Isotropic Assumption)
        mt = magnitude_to_moment_tensor_isotropic(src_mag)  # Shape (6,)

        # 3. Build Input Matrix (N_sensors x 11 features)
        # Columns: [Lat, Lon, Depth, Dist_km, Variance, m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]
        # We need to tile the source-specific features to match N sensors
        N = len(sensors)

        # Create the feature matrix
        features = np.zeros((N, 11))

        # Fill sensor-specific columns
        # Note: Model trained on source-relative coords?
        # Usually ML models take raw coords if trained globally.
        # Using Source Lat/Lon here per your csv sample (row 1 implies these are inputs).
        # WAIT: Your CSV has "Lat", "Lon" columns. Are these Source or Sensor?
        # Based on "Distance_to_source_km" being a separate col, "Lat/Lon" are likely RECEIVER coords.
        features[:, 0] = sens_lat
        features[:, 1] = sens_lon

        # Fill Source-specific columns (repeated for all sensors)
        features[:, 2] = src_depth
        features[:, 3] = dists_km
        features[:, 4] = sens_fidelity

        # Fill Moment Tensor (repeated)
        features[:, 5:11] = np.tile(mt, (N, 1))

        # 4. Scale Inputs
        features_scaled = self.x_scaler.transform(features).astype(np.float32)
        if not hasattr(self, "_printed_debug"):
            print(
                f"[ML power debug] raw_depth_feature={features[0,2]:.3f}, scaled_depth_feature={self.x_scaler.transform(features[:1])[0,2]:.3f}, dist_km={features[0,3]:.3f}, fidelity={features[0,4]:.3f}"
            )
            print("[ML debug raw]", features[0])
            print("[ML debug z]", self.x_scaler.transform(features[:1])[0])
            self._printed_debug = True

        # 5. Inference
        with torch.no_grad():
            tensor_x = torch.from_numpy(features_scaled).to(self.device)
            pred_std = self.model(tensor_x).view(-1).cpu().numpy()

        # 6. Inverse Transform Outputs
        # y_scaler expects (N, 1)
        log_power_pred = self.y_scaler.inverse_transform(
            pred_std.reshape(-1, 1)
        ).ravel()

        return log_power_pred


# Global instance variable (to be initialized by main code)
_POWER_MODEL = None


def get_power_model(model_path=None, x_path=None, y_path=None):
    """Singleton accessor to avoid reloading weights."""
    global _POWER_MODEL
    if _POWER_MODEL is None:
        if model_path is None:
            raise ValueError("Must provide paths for first initialization")
        _POWER_MODEL = SeismicPowerInterface(model_path, x_path, y_path)
    return _POWER_MODEL
