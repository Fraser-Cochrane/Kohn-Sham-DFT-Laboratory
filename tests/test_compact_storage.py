"""Regression tests for the compact DFT runtime and cache representation."""

from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import atomic_orbital_master as model


class CompactStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.original_cache_directory = model.CACHE_DIRECTORY
        model.CACHE_DIRECTORY = Path(self.temporary_directory.name) / "cache"

    def tearDown(self) -> None:
        model.CACHE_DIRECTORY = self.original_cache_directory
        self.temporary_directory.cleanup()

    def test_all_elements_use_pyscf_bundled_bases(self) -> None:
        failures = []
        for symbol, atomic_number in model.ATOMIC_NUMBERS.items():
            try:
                model.select_orbital_basis(symbol, atomic_number)
            except Exception as exc:  # pragma: no cover - assertion reports detail
                failures.append((symbol, type(exc).__name__, str(exc)))
        self.assertEqual(failures, [])

    def test_array_bundles_are_compressed_and_round_trip(self) -> None:
        path = model.cache_bundle("spatial-grids", "round-trip")
        source = np.zeros((32, 32, 32), dtype=np.float32)
        source[8:24, 8:24, 8:24] = 1.0
        self.assertTrue(
            model.atomic_save_array_bundle(
                path,
                {"density": source},
                {"purpose": "test"},
            )
        )
        self.assertTrue((path / "density.npz").is_file())
        self.assertFalse((path / "density.npy").exists())
        loaded = model.load_array_bundle(path, ("density",))
        self.assertIsNotNone(loaded)
        arrays, metadata = loaded
        np.testing.assert_array_equal(arrays["density"], source)
        self.assertEqual(metadata["purpose"], "test")

    def test_obsolete_cache_layout_is_removed_safely(self) -> None:
        root = model.CACHE_DIRECTORY
        obsolete = root / "rendered-results"
        obsolete.mkdir(parents=True)
        (obsolete / "old.html").write_text("obsolete", encoding="utf-8")
        unrelated = root / "user-file.txt"
        unrelated.write_text("preserve", encoding="utf-8")

        self.assertEqual(model.cache_root(), root.resolve())
        self.assertFalse(obsolete.exists())
        self.assertEqual(unrelated.read_text(encoding="utf-8"), "preserve")
        self.assertEqual(
            (root / ".cache-format-version").read_text(encoding="ascii"),
            str(model.CACHE_FORMAT_VERSION),
        )

    def test_pymcubes_surface_preserves_coordinate_scale(self) -> None:
        axis = np.linspace(-2.0, 2.0, 41)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        density = np.asarray(x * x + y * y + z * z, dtype=np.float32)
        vertices, faces = model.extract_isosurface_mesh(
            density,
            1.0,
            float(axis[1] - axis[0]),
        )
        vertices += axis[0]
        radii = np.linalg.norm(vertices, axis=1)
        self.assertEqual(vertices.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)
        self.assertLess(abs(float(np.mean(radii)) - 1.0), 0.01)
        self.assertLess(float(np.max(np.abs(radii - 1.0))), 0.02)

    def test_lazy_plot_json_uses_gzip_storage(self) -> None:
        path = Path(self.temporary_directory.name) / "figure.json.gz"
        text = '{"data":[],"layout":{}}'
        model.atomic_write_gzip_text(path, text)
        self.assertEqual(model.read_cached_figure_json(path), text)
        with gzip.open(path, mode="rt", encoding="utf-8") as handle:
            self.assertEqual(handle.read(), text)

    def test_result_page_reuses_shared_plotly_asset(self) -> None:
        output = Path(self.temporary_directory.name) / "result.html"
        figure = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
        result = model.write_results_website(
            {"radial": ("Radial", "Test figure", figure)},
            -1.0,
            -0.5,
            0.9,
            {
                "basis": "test-basis",
                "hamiltonian": "test",
                "radial_solver": "test",
                "occupations": "test",
            },
            output,
            "test",
            0.1,
            "0" * 64,
        )
        html = result.read_text(encoding="utf-8")
        self.assertIn('<script src="../assets/plotly.min.js"></script>', html)
        self.assertNotIn("plotly.js v", html)
        # The first Plotly figure remains in the page, while the 4+ MiB Plotly
        # runtime is shared by every result instead of copied into this file.
        self.assertLess(result.stat().st_size, 250_000)

    def test_web_routes_serve_home_and_shared_plotly(self) -> None:
        client = model.app.test_client()
        home = client.get("/")
        asset = client.get("/assets/plotly.min.js")
        self.assertEqual(home.status_code, 200)
        self.assertEqual(asset.status_code, 200)
        self.assertGreater(len(asset.data), 1_000_000)
        self.assertIn("max-age=86400", asset.headers["Cache-Control"])


if __name__ == "__main__":
    unittest.main()
