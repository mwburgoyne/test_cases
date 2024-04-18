import numpy as np

# Empirical Density and Viscosity of pure CO2, fit to NIST isochore data calculated with Span-Wagner EOS
# Unpublished approach created by M. Burgoyne, November 2023
# Valid over range 0-300 degF and 14.7-15,000 psia

class CO2_props:
    def __init__(self, psia, degf, verbose = False):
        self.psia = psia
        self.degf = degf
        
        self.p_sc, self.t_sc, self.den_sc = 14.696, 60, 0.11663
        self.R = 10.7316
        self.degF2degR = 459.67
        self.pcrit, self.tcrit, self.den_crit, self.mwCO2 = 1070.0, 87.7608, 29.1913, 44.0095
        self.z_crit = (self.pcrit * self.mwCO2) / self.den_crit / self.R / ((self.degF2degR + self.tcrit))
        self.psat, self.tsat = None, None
        if psia <= self.pcrit:
            self.tsat = self.sat_t(psia)
        if degf <= self.tcrit:
            self.psat = self.sat_p(degf)
            
        self.liq, self.vap, self.hp_liq, self.ht_vap, self.supercrit = [False for x in range(5)]
        if psia > self.pcrit and degf > self.tcrit:             # Supercritical
            self.phase, self.supercrit = 'Supercritical', True
        elif psia > self.pcrit:                                 # High pressure liquid
            self.phase, self.hp_liq = 'HP Liquid', True
        elif degf > self.tcrit:                                 # High temperature vapor
            self.phase, self.ht_vap = 'HT Vapor', True
        elif psia > self.psat:                                  # Liquid
            self.phase, self.liq = 'Liquid', True
        else:                                                   # Vapor
            self.phase, self.vap = 'Vapor', True

        self.a_den = [[-8.226208730489812e-21, 3.7438583228404745e-18, -7.439073989837425e-16, 8.498442470036636e-14, -6.183384681302112e-12, 2.9928158015351095e-10, -9.72945997506213e-09, 2.0830205070018375e-07, -2.8033162403081946e-06, 2.2598358237462982e-05, -0.00010613516872445148, 0.0001268035342185783, -2.0532582087597015e-05],
        [1.796472528333071e-18, -8.166345569246089e-16, 1.6326094568592355e-13, -1.8963596975134925e-11, 1.4212504200339453e-09, -7.177786193360776e-08, 2.4541513924892683e-06, -5.511778385193979e-05, 0.0007724235390716527, -0.006609950290463696, 0.04356682541695601, 0.20538364302280873, 0.0062539362431206355],
        [-3.1155604329801795e-17, 1.221129449890288e-14, -2.292546542849714e-12, 2.8869686148020653e-10, -2.695101324921927e-08, 1.8051500020370706e-06, -7.986367109497906e-05, 0.0021615848517313346, -0.0345945557385573, 0.4244409503546505, -8.577746586114094, 114.31682682453885, -0.36688089582770955]]
    
        self.verbose = verbose
        self.p_coefics(degf)
        self.DEN_CO2(degf, psia)
        self.Z = psia * self.mwCO2 / (self.den * self.R * (self.degF2degR + degf))
        self.VIS_CO2(self.den, degf)
        self.calc_cf()
        
        
    # Returns saturation pressure (psia) at a given temperature < tcrit
    def sat_p(self, degf):
        if degf > self.tcrit:
            return None
        a = [2.79565E-11,4.14338E-10,-5.83136E-08,8.25699E-05,0.031904075,5.140738371,305.6732077]
        return sum([a[i]*degf**(len(a)-1-i) for i in range(len(a))])
    
    # Returns saturation temperature (degF) at given pressure < pcrit
    def sat_t(self, psia):
        a = [-7.11504E-16, 2.85753E-12, -4.67342E-09, 4.03906E-06, -0.002047026, 0.731890144, -113.9767372]
        return sum([a[i]*psia**(len(a)-1-i) for i in range(len(a))])
    
    # Returns tuple of saturated vapor and liquid Z-Factors for a temperature < tcrit
    def Sat_Z(self, degf):
        if degf > self.tcrit:
            return (None, None)
        A, B, C, D = 2.265835E-01, -2.581854E-03, 4.527636E-01, 2.743300E-01
        correction_coeffics = [6.0047E-15, -1.86524E-13, -4.97134E-11, 1.25355E-09, 1.21114E-07, -5.09325E-06, 3.4861E-05, 1.002422952]
        correction = sum([correction_coeffics[i]*degf**(len(correction_coeffics)-1-i) for i in range(len(correction_coeffics))])
        GZ = correction*(max((A+B*degf),0)**C+D)
        
        # Saturated Liquid Z-Factor
        LZ_coeffics = [9.613075465, -37.58426547, 60.53155739, -51.27927428, 24.29737599, -6.596439652, 1.022715073]
        LZ = sum([LZ_coeffics[i]*GZ**(len(LZ_coeffics)-1-i) for i in range(len(LZ_coeffics))])
        
        return np.array([GZ, LZ])
    
    # Returns tuple of saturated vapor and liquid densities for a temperature < tcrit
    def Z_Den(self, psat_psia, tsat_degF):
        if tsat_degF > self.tcrit:
            return (None, None)
        return psat_psia * self.mwCO2 / (self.Sat_Z(tsat_degF) * self.R * (self.degF2degR + tsat_degF))
        
    # Returns pressure from density and temperature from relationships fit to isochoric data points
    # Fit over 0 - 300 degF and 14.7 -15,000 psia
    # Quadratic Pressure = Fn(Temperature) relationships fit over many isochore density lines
    # Coefficients for pressure relationship then fit as a function of density
    def p_Fn_Dens(self, den, degf):
        ais = []
        for j in range(len(a)):
            ais.append(sum([self.a_den[j][i]*den**(len(self.a_den[0])-1-i) for i in range(len(self.a_den[0]))]))
        return sum([ais[i]*degf**(len(ais)-1-i) for i in range(len(ais))])
    
    def p_coefics(self, degf):
        den_poly_len = len(self.a_den[0])
        degf_pol_len = len(self.a_den)
        p = [0 for i in range(den_poly_len)]
        for j in range(den_poly_len):
            for i in range(degf_pol_len):
                p[j] += self.a_den[i][j]*degf**(degf_pol_len-i-1)
        self.p_coefic = np.array(p)
        return 

    def density_roots(self, degf, ptarg):
        p = self.p_coefic
        p[-1] += -ptarg
        roots = np.roots(p)
        roots = [np.real(root) for root in roots if np.imag(root) == 0]
        return np.array(roots)
    
    # Pressure at critical density - Fit against isochore originating from critical pressure & temperature (29.1913 lb/cuft)
    def p_at_den_crit(self, degf):
        a = [-2.01664E-06, 0.00109147, 14.04560166, -174.5586692]
        return sum([a[i]*degf**(len(a)-1-i) for i in range(len(a))])
    
    # Returns CO2 density (lb/cuft) at specified pressure and temperature
    def DEN_CO2(self, degf, psia):
        den_err = 0 #0.25
        max_den, min_den = 76.657, 0.070189 # Max and min for 0 degF x 15,000 psia and 300 degF x 14.696 psia
        
        # exclude non physical roots
        if self.supercrit: # Supercritical
            p_at_crit_den = self.p_at_den_crit(degf)
            
            if psia > p_at_crit_den:
                min_den, max_den = self.den_crit - den_err, max_den + den_err
            else:
                # 6.480063510980938 lb/cuft is the density at 300 degF and Pcrit (ie lower limit density for supercritical)
                min_den, max_den = 6.480063510980938 - den_err, self.den_crit + den_err
        else:
            if self.hp_liq:              # High pressure liquid
                min_den, max_den = self.den_crit - den_err, max_den + den_err
            if self.ht_vap:            # High Temperature vapor
                min_den, max_den = min_den, self.den_crit + den_err
            if self.vap or self.liq:
                Sat_GDEN, Sat_LDEN = self.Z_Den(self.psat, self.degf)
                if self.vap:         # Vapor
                    min_den, max_den = min_den, Sat_GDEN
                else:                   # Liquid
                    min_den, max_den = Sat_LDEN - den_err, max_den
                    
        # Using inbuilt Numpy function to find roots of the polynomial with coefficients calculated by density_roots()
        # If doing this in Excel, you'd just need to minimize the absolute value of that polynomial between 
        # the min/max density limits using Netwon / Bisection methods. 
        # The slope of the error function can easily be calculated analytically since its a polynomial.
        roots = self.density_roots(degf, psia)
        
        if self.verbose:
            print('Limits:', min_den, max_den)
            print(roots, self.phase, psia, degf)
        
        orig_roots = roots
        roots = [root for root in roots if root >= min_den and root <= max_den]
        
        if self.verbose:
            print('Roots after filtering:', roots)
        
        if len(roots) == 1:
                self.den = roots[0]
                return
            
        if len(roots) > 1:
            if self.supercrit:
                self.den = sum(roots)/len(roots) # This shouldnt be needed... but just in case to prevent error
                return
            if self.hp_liq or self.liq:
                self.den =  max(roots)
                return
            if self.vap or self.ht_vap:
                self.den =  min(roots)
                return
        
        # Otherwise we have zero roots because the solution was just outside a limit
        # Revisit orignal roots and find the closest one to the limit
        
        maxlim_root_errs  = orig_roots - max_den
        minlim_root_errs = orig_roots - min_den
        
        maxlim_min = min(np.abs(maxlim_root_errs))
        minlim_min = min(np.abs(minlim_root_errs))
        
        if maxlim_min < minlim_min: # Original root was closest to the maximum value
            root_idx = np.where(np.abs(maxlim_root_errs) == maxlim_min)
        else:
            root_idx = np.where(np.abs(minlim_root_errs) == minlim_min)
        self.den = float(orig_roots[root_idx])
        return
      
    # Reciprocal viscosity fit to a polynomial of density with temperature dependant coefficients
    # Empirical fit by M. Burgoyne, November 2023
    def VIS_CO2(self, den, degf):
        b = [[-3.29747693e-23, 4.51642772e-20, -2.69284562e-17, 9.44629519e-15, -2.15903211e-12, 2.69115188e-10],
         [6.82565651e-21, -9.29953499e-18, 5.55108615e-15, -1.95180264e-12, 4.49995123e-10, -5.88101149e-08],
         [-5.07738457e-19, 6.90391036e-16, -4.12832384e-13, 1.45376858e-10, -3.37227687e-08, 4.66626812e-06],
         [1.58107612e-17, -2.14953535e-14, 1.28059783e-11, -4.46483950e-09, 1.01833988e-06, -1.46179530e-04],
         [-1.57877352e-16, 2.14170286e-13, -1.22048313e-10, 3.78724172e-08, -6.17749886e-06, 4.20402423e-05],
         [-4.32595337e-16, 7.14183095e-13, -5.99986552e-10, 3.46211708e-07, -1.63361316e-04, 7.78411048e-02]]
        
        def vis_a_coefics(b, degf):
            ais = [sum([b[j][i]*degf**(len(b[j])-1-i) for i in range(len(b[j]))]) for j in range(len(b))]
            return ais
        
        ai = vis_a_coefics(b, degf)
        One_over_vis = sum([ai[i]*den**(len(ai)-1-i) for i in range(len(ai))])
        self.vis = 1/One_over_vis/1000 # cP        
        
    # Calculates isothermal CO2 compressibility (1/psi)
    def calc_cf(self):
        p = self.p_coefic                                    # Coefficients of P = Fn(Dens)
        dp = [(len(p)-i-1) * p[i] for i in range(len(p)-1)]  # Coefficients of dP/dDens
        dpdV = sum([-dp[i]/(1/self.den)**(len(dp)-i+1) for i in range(len(dp))])
        self.cf = - self.den * 1/dpdV

