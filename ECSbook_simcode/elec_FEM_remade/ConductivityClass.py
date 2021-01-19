from dolfin import *


class SigmaClass(UserExpression):
    """ This is the superclass for conductivity. It is used to make the conductivity tensor
    for FEM simulations. See FEniCS information about 'user-defined expressions by subclassing'
    for more information on how this works"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.slice_thickness = 300
        self.a = self.slice_thickness/2.
        self.elec_z = -self.a
        self.sigma_T = 0.3
        self.sigma_E = 1e5
        self.sigma_S = 0.3
        self.kwargs = kwargs

    def return_conductivity(self, x):
        if x[2] <= -self.a:
            return self.sigma_E
        else:
            return self.sigma_T

    def eval(self, value, x):
        """ This function is written like this so I can use return_conductivity() outside the class for
        debugging reasons """
        value[0] = self.return_conductivity(x)

    def value_shape(self):
        return ()

    def return_conductivity_tensor(self, Vs):
        """ This will return the conductivity scalar for
        isotropic tissue. It will be overridden by Anisotropic"""
        self.sigma = Function(Vs)
        self.sigma.interpolate(self)
        return self.sigma

class ConductivityClass:
    """ This is the superclass for conductivity. It is used to make the conductivity tensor
    for FEM simulations. See FEniCS information about 'user-defined expressions by subclassing'
    for more information on how this works"""
    
    def __init__(self, slice_thickness=300., slice_R=None, sigma_S=1.5, sigma_T=1.5, **kwargs):
        self.slice_thickness = slice_thickness
        self.a = self.slice_thickness/2.
        self.elec_z = -self.a
        self.sigma_T = sigma_T
        self.sigma_S = sigma_S
        self.kwargs = kwargs

    def return_conductivity_tensor(self, Vs):
        """ This will return the conductivity scalar for 
        isotropic tissue. It will be overridden by Anisotropic"""
        self.sigma = Function(Vs)
        self.sigma.interpolate(self)
        return self.sigma


class ControlConductivity(UserExpression, ConductivityClass):
    """ This inherits from ConductivityClass and contains the specific 
    conductivity of the control set-up"""

    def __init__(self, **kwargs):
        ConductivityClass.__init__(self, **kwargs)
        
    def return_conductivity(self, x):
        if x[2] < self.a:
            return self.sigma_T
        else:
            return self.sigma_S

    def eval(self, value, x):
        """ This function is written like this so I can use return_conductivity() outside the class for
        debugging reasons """
        value[0] = self.return_conductivity(x)


class ElectrodeConductivity(UserExpression, ConductivityClass):
    """ This inherits from ConductivityClass and contains the specific 
    conductivity of the control set-up"""

    def __init__(self, **kwargs):

        ConductivityClass.__init__(self, **kwargs)

    def return_conductivity(self, x):

        if -self.a < x[2] < self.a:
            return self.sigma_T
        elif x[2] <= -self.a:
            return self.kwargs['sigma_E']
        else:
            return self.sigma_S

    def eval(self, value, x):
        """ This function is written like this so I can use return_conductivity() outside the class for
        debugging reasons """
        value[0] = self.return_conductivity(x)
        

class SalineLayer(UserExpression, ConductivityClass):
    """ This inherits from ConductivityClass and contains the specific 
    conductivity of the set-up with a thin saline interface"""

    def __init__(self, **kwargs):
        ConductivityClass.__init__(self, **kwargs)
        
        if not 'interface_thickness' in kwargs:
            raise RuntimeError("Conductivity not initalized right. Need interface_thickness")

    def return_conductivity(self, x):
        if (-self.a + self.kwargs['interface_thickness']) < x[2] < self.a:
            return self.sigma_T
        else:
            return self.sigma_S

    def eval(self, value, x):
        """ This function is written like this so I can use return_
        conductivity() outside the class for
        debugging reasons """
        value[0] = self.return_conductivity(x)

class SalineLayerSimplified(UserExpression, ConductivityClass):
    """ This inherits from ConductivityClass and contains the specific
    conductivity of the set-up with a thin saline interface"""

    def __init__(self, **kwargs):
        ConductivityClass.__init__(self, **kwargs)

        if not 'interface_thickness' in kwargs:
            raise RuntimeError("Conductivity not initalized right. Need interface_thickness")

    def return_conductivity(self, x):
        if (-self.a + self.kwargs['interface_thickness']) > x[2]:
            return self.sigma_T
        else:
            return self.sigma_S

    def eval(self, value, x):
        """ This function is written like this so I can use return_
        conductivity() outside the class for
        debugging reasons """
        value[0] = self.return_conductivity(x)



class Obstacle(UserExpression, ConductivityClass):
    """ This inherits from ConductivityClass and contains the specific
    conductivity of the set-up with a thin saline interface"""

    def __init__(self, **kwargs):
        ConductivityClass.__init__(self, **kwargs)

        if not 'interface_thickness' in kwargs:
            raise RuntimeError("Conductivity not initalized right. Need interface_thickness")

    def return_conductivity(self, x):
        if x[2] > self.a:
            return self.sigma_S
        elif -self.a/1.5 < x[2] < -self.a/3. and np.sqrt(x[0]**2 + x[1]**2) < 200:
            return self.sigma_T / 3.
        else:
            return self.sigma_T

    def eval(self, value, x):
        """ This function is written like this so I can use return_
        conductivity() outside the class for
        debugging reasons """
        value[0] = self.return_conductivity(x)


class InfHomoConductivity(UserExpression, ConductivityClass):
    """ This inherits from ConductivityClass and contains the specific 
    conductivity of the infinite homogeneous (no saline) set-up"""

    def __init__(self, **kwargs):
        ConductivityClass.__init__(self, **kwargs)
        
    def return_conductivity(self, x):
        return self.sigma_T

    def eval(self, value, x):
        """ This function is written like this so I can use return_conductivity() outside the class for
        debugging reasons """
        value[0] = self.return_conductivity(x)

        
class Anisotropic(ConductivityClass):
    """ This inherits from ConductivityClass and contains the specific 
    conductivity of the anisotropic set-up"""

    def __init__(self, **kwargs):
        ConductivityClass.__init__(self, **kwargs)

        _set_anis_x = {
            'sigma_T': self.sigma_T[0],
            'sigma_S': self.sigma_S,
            'slice_thickness': self.slice_thickness,
            }

        _set_anis_y = {
            'sigma_T': self.sigma_T[1],
            'sigma_S': self.sigma_S,
            'slice_thickness': self.slice_thickness,
            }
        
        _set_anis_z = {
            'sigma_T': self.sigma_T[2],
            'sigma_S': self.sigma_S,
            'slice_thickness': self.slice_thickness,
            }

        self._sigma_x = ControlConductivity(**_set_anis_x)
        self._sigma_y = ControlConductivity(**_set_anis_y)
        self._sigma_z = ControlConductivity(**_set_anis_z)


    def return_conductivity_tensor(self, Vs):
        """ This will return the conductivity scalar for 
        isotropic tissue. It overrides the default in ConductivityClass """
        self.sigma_x = Function(Vs)
        self.sigma_x.interpolate(self._sigma_x)
        self.sigma_y = Function(Vs)
        self.sigma_y.interpolate(self._sigma_y)
        self.sigma_z = Function(Vs)
        self.sigma_z.interpolate(self._sigma_z)

        self.sigma = as_matrix(((self.sigma_x, Constant(0), Constant(0)),
                                (Constant(0), self.sigma_y, Constant(0)),
                                (Constant(0), Constant(0), self.sigma_z)))
        return self.sigma
        
        
class Inhomogeneous(UserExpression, ConductivityClass):
    """ This inherits from ConductivityClass and contains the specific 
    conductivity of the inhomogeneous set-up"""

    def __init__(self, **kwargs):
        ConductivityClass.__init__(self, **kwargs)
        
    def return_conductivity(self, x):
        if x[2] < self.a:
            if x[0] <= self.kwargs['inhomo_x_pos']:
                return self.kwargs['sigma_T1']
            else:
                return self.kwargs['sigma_T2']
        else:
            return self.sigma_S

    def eval(self, value, x):
        """ This function is written like this so I can use return_conductivity() outside the class for
        debugging reasons """
        value[0] = self.return_conductivity(x)
