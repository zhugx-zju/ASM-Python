import unittest

import numpy as np

from fgm_asm import MeshInfo, generate_fgm_modulus, generate_grf_field


class GrfGenerationTests(unittest.TestCase):
    def test_generation_is_deterministic_and_bounded(self):
        mesh = MeshInfo(9.0, 9.0, 4, 3)
        field_a = generate_grf_field(
            mesh, E_max=8.0, sigma_g=1.0, ell=1.0, seed=17
        )
        field_b = generate_grf_field(
            mesh, E_max=8.0, sigma_g=1.0, ell=1.0, seed=17
        )

        self.assertEqual(field_a.shape, (mesh.nods_y, mesh.nods_x))
        np.testing.assert_array_equal(field_a, field_b)
        self.assertTrue(np.all(field_a >= 0.0))
        self.assertLessEqual(float(np.max(field_a)), 8.0)

    def test_single_sample_uses_configured_maximum(self):
        mesh = MeshInfo(9.0, 9.0, 4, 4)
        field, _ = generate_fgm_modulus(
            mesh,
            dis_type="grf",
            grf_E_max=8.0,
            grf_sigma_g=1.0,
            grf_ell=1.0,
            grf_seed=5,
        )

        self.assertEqual(field.shape, (mesh.nods_y, mesh.nods_x))
        self.assertGreater(float(np.max(field)), 1.0)
        self.assertLessEqual(float(np.max(field)), 8.0)


if __name__ == "__main__":
    unittest.main()
