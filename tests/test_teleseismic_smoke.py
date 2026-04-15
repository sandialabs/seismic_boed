import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spatial_domain import SpatialDomain


def _have_optional_deps():
    required = ["mpi4py", "obspy", "sklearn"]
    missing = []
    for module_name in required:
        try:
            __import__(module_name)
        except Exception:
            missing.append(module_name)
    return missing


@unittest.skipIf(bool(_have_optional_deps()), f"Missing optional smoke-test deps: {_have_optional_deps()}")
class TeleseismicSmokeTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_eig_calc_spherical_globe_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            bounds_path = tmpdir / "globe_bounds.json"
            bounds_path.write_text(
                json.dumps(
                    {
                        "geometry_mode": "spherical",
                        "domain_type": "globe",
                        "depth_range": [5.0, 15.0],
                        "mag_range": [4.5, 5.5],
                    }
                ),
                encoding="utf-8",
            )

            input_path = tmpdir / "eig_inputs.dat"
            input_path.write_text(
                "\n".join(
                    [
                        "2",
                        "2",
                        "1",
                        str(bounds_path),
                        "uniform_prior.py",
                        "0.0,0.0,0.1,2,0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, "eig_calc.py", str(input_path), str(tmpdir / "out.npz"), "0"],
                cwd=self.repo_root,
                env={**os.environ, "SEISMIC_OED_USE_MT_POWER": "0"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_network_opt_spherical_cap_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            event_bounds_path = tmpdir / "event_cap_bounds.json"
            event_bounds_path.write_text(
                json.dumps(
                    {
                        "geometry_mode": "spherical",
                        "domain_type": "spherical_cap",
                        "center_lat": 20.0,
                        "center_lon": -150.0,
                        "radius_deg": 12.0,
                        "depth_range": [5.0, 15.0],
                        "mag_range": [4.5, 5.5],
                    }
                ),
                encoding="utf-8",
            )
            sensor_bounds_path = tmpdir / "sensor_cap_bounds.json"
            sensor_bounds_path.write_text(
                json.dumps(
                    {
                        "geometry_mode": "spherical",
                        "domain_type": "spherical_cap",
                        "center_lat": 20.0,
                        "center_lon": -150.0,
                        "radius_deg": 10.0,
                    }
                ),
                encoding="utf-8",
            )

            opt_input_path = tmpdir / "inputs_opt.dat"
            opt_input_path.write_text(
                "\n".join(
                    [
                        "1",
                        "1",
                        str(sensor_bounds_path),
                        "2,2,0",
                        "0",
                        "2",
                        "2",
                        "1",
                        str(event_bounds_path),
                        "",
                        "uniform_prior.py",
                        "1",
                        "20.0,-150.0,0.1,2,0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            save_dir = tmpdir / "opt_out"
            result = subprocess.run(
                [
                    sys.executable,
                    "network_opt.py",
                    str(opt_input_path),
                    "network_result.npz",
                    str(save_dir),
                    "0",
                ],
                cwd=self.repo_root,
                env={**os.environ, "SEISMIC_OED_USE_MT_POWER": "0"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            result_path = save_dir / "network_result.npz"
            self.assertTrue(result_path.exists())

            sensors = np.load(result_path)["sensors"]
            cap_domain = SpatialDomain.from_file(str(sensor_bounds_path), sensor_bounds=True)
            self.assertTrue(cap_domain.contains(sensors[:, :2]).all())


if __name__ == "__main__":
    unittest.main()
