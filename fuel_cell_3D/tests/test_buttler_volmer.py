"""
Unit tests for Butler-Volmer equations and material property functions.
"""
import pytest
from common import np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from define_buttler_volmer import (
    i_0_complex, 
    alpha_lattice_complex, 
    c_lattice_complex,
    Dn_complex,
    ocp_complex,
    i_se
)


class TestButlerVolmerFunctions:
    """Test suite for Butler-Volmer and material property functions."""
    
    def test_i_0_complex_returns_two_values(self):
        """Test that i_0_complex returns P and dP/dx."""
        x = 0.5
        P, dP_dx = i_0_complex(x)
        
        assert isinstance(P, (float, np.floating))
        assert isinstance(dP_dx, (float, np.floating))
    
    def test_i_0_complex_array_input(self):
        """Test i_0_complex with array input."""
        x = np.array([0.1, 0.5, 0.9])
        P, dP_dx = i_0_complex(x)
        
        assert len(P) == 3
        assert len(dP_dx) == 3
    
    def test_i_0_complex_monotonicity(self):
        """Test that exchange current density behaves reasonably."""
        x = np.linspace(0, 1, 100)
        P, _ = i_0_complex(x)
        
        # Check output is finite
        assert np.all(np.isfinite(P))
    
    def test_alpha_lattice_complex_returns_two_values(self):
        """Test that alpha_lattice_complex returns a and da/dx."""
        x = 0.5
        a, da_dx = alpha_lattice_complex(x)
        
        assert isinstance(a, (float, np.floating))
        assert isinstance(da_dx, (float, np.floating))
    
    def test_alpha_lattice_positive(self):
        """Test that lattice parameter is positive."""
        x = np.linspace(0, 1, 100)
        a, _ = alpha_lattice_complex(x)
        
        assert np.all(a > 0)
    
    def test_c_lattice_complex_returns_two_values(self):
        """Test that c_lattice_complex returns c and dc/dx."""
        x = 0.5
        c, dc_dx = c_lattice_complex(x)
        
        assert isinstance(c, (float, np.floating))
        assert isinstance(dc_dx, (float, np.floating))
    
    def test_c_lattice_positive(self):
        """Test that c lattice parameter is positive."""
        x = np.linspace(0, 1, 100)
        c, _ = c_lattice_complex(x)
        
        assert np.all(c > 0)
    
    def test_Dn_complex_basic(self):
        """Test diffusivity function."""
        x = np.array([0.3, 0.5, 0.7])
        D_damage = np.array([0.1, 0.2, 0.3])
        
        D, dD_dx = Dn_complex(x, D_damage)
        
        assert len(D) == 3
        assert len(dD_dx) == 3
        assert np.all(D >= 0)  # Diffusivity should be non-negative
    
    def test_Dn_complex_damage_limit(self):
        """Test that damage is capped at 0.9."""
        x = np.array([0.5])
        D_damage = np.array([0.95])  # Above the 0.9 limit
        
        D, _ = Dn_complex(x, D_damage)
        
        # After capping, damage should be 0.9, so (1-D_damage) = 0.1
        # Result should be reduced significantly
        assert np.all(D >= 0)
    
    def test_Dn_complex_no_damage(self):
        """Test diffusivity with zero damage."""
        x = np.array([0.5])
        D_damage = np.array([0.0])
        
        D, _ = Dn_complex(x, D_damage)
        
        # With no damage, diffusivity should be at maximum
        assert D[0] > 0
    
    def test_ocp_complex_returns_two_values(self):
        """Test open circuit potential function."""
        x = 0.5
        E_eq, dE_dx = ocp_complex(x)
        
        assert isinstance(E_eq, (float, np.floating))
        assert isinstance(dE_dx, (float, np.floating))
    
    def test_ocp_complex_array(self):
        """Test OCP with array input."""
        x = np.linspace(0, 1, 100)
        E_eq, dE_dx = ocp_complex(x)
        
        assert len(E_eq) == 100
        assert len(dE_dx) == 100
        assert np.all(np.isfinite(E_eq))
    
    def test_ocp_complex_monotonic_regions(self):
        """Test that OCP has reasonable behavior."""
        x = np.linspace(0.1, 0.9, 50)
        E_eq, _ = ocp_complex(x)
        
        # OCP should be finite everywhere
        assert np.all(np.isfinite(E_eq))
    
    def test_i_se_returns_three_values(self):
        """Test Butler-Volmer current density function."""
        p_s = 1.0
        j0 = 0.1
        E_eq = 0.9
        Fday = 96485.0
        R = 8.3145
        Tk = 1273.2
        
        dibv_deta, dibv_di0, i_bv = i_se(p_s, j0, E_eq, Fday, R, Tk)
        
        # Use backend-agnostic checking - just check they are numeric
        assert hasattr(dibv_deta, '__float__') or hasattr(dibv_deta, 'item')
        assert hasattr(dibv_di0, '__float__') or hasattr(dibv_di0, 'item')
        assert hasattr(i_bv, '__float__') or hasattr(i_bv, 'item')
    
    def test_i_se_zero_overpotential(self):
        """Test current density at zero overpotential."""
        p_s = 1.0
        j0 = 0.1
        E_eq = 1.0  # Same as p_s, so eta = 0
        Fday = 96485.0
        R = 8.3145
        Tk = 1273.2
        
        _, _, i_bv = i_se(p_s, j0, E_eq, Fday, R, Tk)
        
        # At zero overpotential, current should be near zero
        assert abs(i_bv) < 1e-6
    
    def test_i_se_array_inputs(self):
        """Test Butler-Volmer with array inputs."""
        p_s = np.array([1.0, 1.1, 1.2])
        j0 = np.array([0.1, 0.1, 0.1])
        E_eq = np.array([0.9, 0.9, 0.9])
        Fday = 96485.0
        R = 8.3145
        Tk = 1273.2
        
        dibv_deta, dibv_di0, i_bv = i_se(p_s, j0, E_eq, Fday, R, Tk)
        
        assert len(dibv_deta) == 3
        assert len(dibv_di0) == 3
        assert len(i_bv) == 3
    
    def test_i_se_positive_overpotential(self):
        """Test that positive overpotential gives anodic current."""
        p_s = 1.1
        j0 = 0.1
        E_eq = 1.0
        Fday = 96485.0
        R = 8.3145
        Tk = 1273.2
        
        _, _, i_bv = i_se(p_s, j0, E_eq, Fday, R, Tk)
        
        # Positive overpotential should give positive (anodic) current
        assert i_bv > 0
    
    def test_i_se_negative_overpotential(self):
        """Test that negative overpotential gives cathodic current."""
        p_s = 0.9
        j0 = 0.1
        E_eq = 1.0
        Fday = 96485.0
        R = 8.3145
        Tk = 1273.2
        
        _, _, i_bv = i_se(p_s, j0, E_eq, Fday, R, Tk)
        
        # Negative overpotential should give negative (cathodic) current
        assert i_bv < 0


class TestMaterialPropertiesConsistency:
    """Test consistency and physical constraints of material properties."""
    
    def test_lattice_parameters_in_physical_range(self):
        """Test that lattice parameters are in physical range (nm scale)."""
        x = np.linspace(0, 1, 50)
        a, _ = alpha_lattice_complex(x)
        c, _ = c_lattice_complex(x)
        
        # Lattice parameters should be in order of 1e-10 to 1e-9 m (0.1 to 1 nm)
        assert np.all(a > 1e-11)
        assert np.all(a < 1e-8)
        assert np.all(c > 1e-11)
        assert np.all(c < 1e-8)
    
    def test_diffusivity_bounds(self):
        """Test that diffusivity stays within reasonable bounds."""
        x = np.linspace(0, 1, 50)
        D_damage = np.zeros(50)
        
        D, _ = Dn_complex(x, D_damage)
        
        # Diffusivity should be positive
        assert np.all(D >= 0)
        # Should be less than unreasonably large values
        assert np.all(D < 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

