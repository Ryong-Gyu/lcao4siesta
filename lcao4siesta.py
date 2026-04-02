####                                       ####
#                                             #
#   SEISTA Oribtal Projected Module by RONG   #
#                                             #
####                                       ####

# Ver. 1.0. Date : 2020/6/17

#--------- Medules list ---------#

import siesta_io as io
import numpy as np
import scipy
import glob
import math
import time
from scipy.interpolate import interp1d



#------- Parameters -------#

pi = np.pi
bohr2ang = 0.529177249
smearing = 0.2
overlap_tolerance = 1.0e-10
phi_tolerance = 1.0e-10

class lcao:
    
    def __init__(self, system='Carbon', dm_file=None, ion_files=None):

        #---- Input variables ----#
        self._system = system
        self._dm_file = dm_file if dm_file is not None else self._system + '.DM'
        self._ion_files = ion_files

        #---- Basic information only (requested) ----#
        self._readDM()
        self._readIon()

        #---- Initiate variable ----#
        self._target = []

    def _readDM(self):
        results = io.read_dm(self._dm_file)
        self.dm_nb = results[0]
        self.dm_ns = results[1]
        self.dm_numd = results[2]
        self.dm_listdptr = results[3]
        self.dm_listd = results[4]
        self.dm = results[5]

    #------- Methods : Read and Write (IO) -------#
        
    def _readWFSX(self):
        
        system_label = self._system
        file_name = system_label + '.WFSX'
        results = io.readWFSX(file_name)

        self.gamma = results[0]
        self.kpoints = results[1]
        self.kweight = results[2]
        self.wavefunction = results[3]
        self.eigenvalue = results[4]
        
        self._atom_index = results[5]
        self._atom_symbol = results[6]
        self._orbital_index = results[7]
        self._orbital_n = results[8]
        self._orbital_symbol = results[9]

    def _readHSX(self):
        
        system_label = self._system
        file_name = system_label + '.HSX'
        results = io.readHSX(file_name)    

        self.numh = results[0]
        self.listhptr = results[1]
        self.listh = results[2]
        self.indxuo = results[3]
        
        self.Hamilt = results[4]
        self.Sover = results[5]
        self.xij = results[6]
        
        self.atom_index = results[7]
        self.atom_species = results[8]
        self.orbital_n = results[9]
        self.orbital_l = results[10]
        self.orbital_ml = results[11]   # m + l + 1
        self.orbital_zeta = results[12]

    def _readIon(self):
        
        self.ions = {}
        if self._ion_files is not None:
            for symbol, file_name in self._ion_files.items():
                self.ions.update({symbol: io.read_ion(file_name)})
            return

        for file_name in glob.glob('*.ion'):
            symbol = file_name.replace('.ion', '')
            ion = io.read_ion(file_name)
            self.ions.update({symbol:ion})

    def _ensure_wfsx_hsx(self):
        if not hasattr(self, 'gamma'):
            self._readWFSX()
        if not hasattr(self, 'numh'):
            self._readHSX()

    def _ensure_struct_supercell(self):
        if not hasattr(self, 'cell'):
            self._readStruct()
        if not hasattr(self, '_supercell_vector_list'):
            self._build_supercell_vectors()

    def _readStruct(self):
        
        results = io.readStruct()
        
        self.cell = results[0]
        self.atoms = results[1]
        self.species = results[2]


    #------- Methods : Calculation -------#
            
    def Rnl(self, symbol, n, l, zeta, r):
        
        pao_basis = self.ions[symbol][n][l][zeta]

        cutoff_radius = pao_basis['cutoff']
                    
        if r < cutoff_radius:
            R = pao_basis['r']
            Phi = pao_basis['phi']
            f_phi = interp1d(R,Phi)
            Phi_r = f_phi(r)
        else:
            Phi_r = 0
    
        return Phi_r
                
    def Yml(self, vector, m, l):
        
        x, y, z = vector
         
        r2 = vector**2    
        r = np.sqrt(sum(r2))

        phi = np.arccos(z/r)
        theta = np.arctan2(y,x) + pi

        Y_theta_phi = scipy.special.sph_harm(m, l, theta, phi)
        
        return Y_theta_phi
      
    def delta(self, x):
        
        if (abs(x) > 8*smearing):
            result = 0
        else:
            result = np.exp(-(x/smearing)**2) / (smearing*np.sqrt(pi))
            
        return result
    
    def length(self, vector):
        
        square = 0
        for ix in vector:
            square += ix**2
        return np.sqrt(square)
    
    
    #------- Methods : Orbital Mask -------#
    
    def orbital_mask(self, select): # label, n, l, m, zeta

        '''
        Usage : ['X_n_l_m_zeta'] or ['X_n_l_m'] or ['X_n_l'] or ... all possible 
            
                X : 'C' or '1'  Atomic symbol or Atom's serial number
        
        '''
 #       label, za, zn, zl, zx, zz, 
        
        label = self._atom_symbol
        za = self.atom_index
        zn = self.orbital_n
        zl = self.orbital_l
        zx = self.orbital_ml
        zz = self.orbital_zeta
        
        for index in select:
            
            indexes = index.split('_')
            islabel= 0
            
            try:
                atom_index = int(indexes[0])
            except ValueError:
                atom_label = indexes[0]
                islabel = 1
                
            if (islabel):
                ibuff = np.where(label == atom_label)[0]
            else:
                ibuff = np.where(za == atom_index)[0]
            
            print(ibuff)
            
            if len(indexes) >= 2:
                buff = []
                atom_n = int(indexes[1])
                for i in ibuff:
                    if zn[i] == atom_n:
                        buff.append(i)
                ibuff = buff
                
                print(ibuff)
                
                if len(indexes) >= 3:
                    buff = []
                    atom_l = int(indexes[2])
                    for i in ibuff:
                        if zl[i] == atom_l:
                            buff.append(i)
                    ibuff = buff
                    
                    print(ibuff)
                    
                    if len(indexes) >= 4:
                        buff = []
                        atom_m = int(indexes[3])
                        for i in ibuff:
                            if zx[i] == atom_m + zl[i] + 1:
                                buff.append(i)
                        ibuff = buff
                        
                        print(ibuff)
                        
                        if len(indexes) == 5:
                            buff = []
                            atom_zeta = int(indexes[4])
                            for i in ibuff:
                                if zz[i] == atom_zeta:
                                    buff.append(i)
                            ibuff = buff
                            
                            print(ibuff)
                            
            m_iao = ibuff
            nao = len(m_iao)
            m_ia = np.zeros((nao), dtype = int)
            m_izn = np.zeros((nao), dtype = int)
            m_izl = np.zeros((nao), dtype = int)
            m_izm = np.zeros((nao), dtype = int)
            m_izz = np.zeros((nao), dtype = int)
    
            for i in range(nao):
                index = m_iao[i]
                m_ia[i] = za[index]
                m_izn[i] = zn[index]
                m_izl[i] = zl[index]
                m_izm[i] = zx[index] - zl[index] - 1
                m_izz[i] = zz[index]
            
            buff_info = {
                    'number_of_components' : nao,
                    'atomic_index' : m_ia,
                    'orbital_index' : m_iao,
                    'principle_quantum_number' : m_izn,
                    'angular_quantum_number' : m_izl,
                    'magnetic_quantum_nuber' : m_izm,
                    'zeta' : m_izz
                }
            
            self._target.append(buff_info)
            

    def mask_to_pointer(self):

        target = self._target
        
        numh= self.numh
        listhptr = self.listhptr
        listh = self.listh
        indxuo = self.indxuo
        Sover = self.Sover
        
        list_ptr = []
        list_io = []
    
        for i1 in target:
            list_ptr.append([])
            list_io.append([])
            
            iao_list = i1['orbital_index']
            
            for j1 in iao_list:
                for k1 in range(numh[j1]): 
                    ind = listhptr[j1] + k1
                    io = indxuo[listh[ind]-1] # python <-> fortran index
                    if abs(Sover[ind]) >= overlap_tolerance:
                        list_ptr[-1].append(ind)
                        list_io[-1].append(io-1) # python <-> fortran index
        
        list_ptr = np.array(list_ptr)
        list_io = np.array(list_io)        
        number_of_projections = len(list_ptr[0])
        print('Total number of  prjections : %d'%number_of_projections)
        
        self._projection_ptr = list_ptr
        self._projection_io = list_io

    #------- Methods : Real space grid -------#

    def unit_cell_grid(self, cell, mesh):
        """
        readGrid 형식 입력(cell, mesh)을 받아 rho 형식 좌표 그리드를 리턴.
        Return: (xrho, yrho, zrho) with shape (1, mesh[0], mesh[1], mesh[2]).
        """
        cell = np.array(cell)
        na, nb, nc = int(mesh[0]), int(mesh[1]), int(mesh[2])
        
        if na == 1: ua = cell[0]
        else:       ua = cell[0] / (na-1)
        if nb == 1: ub = cell[1]
        else:       ub = cell[1] / (nb-1)
        if nc == 1: uc = cell[2]
        else:       uc = cell[2] / (nc-1)
        
        xgrid = np.zeros((1, na, nb, nc), dtype = float)
        ygrid = np.zeros((1, na, nb, nc), dtype = float)
        zgrid = np.zeros((1, na, nb, nc), dtype = float)
        
        for ia in range(na):
            for ib in range(nb):
                for ic in range(nc):
                    position_vector = ia * ua + ib * ub + ic * uc
                    xgrid[0, ia,ib,ic] = position_vector[0]
                    ygrid[0, ia,ib,ic] = position_vector[1]
                    zgrid[0, ia,ib,ic] = position_vector[2]
        return xgrid, ygrid, zgrid
        
    def _build_supercell_vectors(self):
        
        cell = self.cell
        basis = self.ions
                
        max_cutoff = 0
        for species in basis:
            for qn in basis[species]:
                for ql in basis[species][qn]:
                    for qlm in basis[species][qn][ql]:
                        cutoff = basis[species][qn][ql][qlm]['cutoff']
                        if cutoff >= max_cutoff:
                            max_cutoff = cutoff
            
        length_a = self.length(cell[0])
        length_b = self.length(cell[1])
        length_c = self.length(cell[2])
        
        na = int(math.ceil(max_cutoff/length_a))
        nb = int(math.ceil(max_cutoff/length_b))
        nc = int(math.ceil(max_cutoff/length_c))
        
        nsc = (2*na+1) * (2*nb+1) * (2*nc+1)
        vectors = np.zeros((nsc,3), dtype = float)
        
        index = 0
        for ia in range(2*na+1):
            for ib in range(2*nb+1):
                for ic in range(2*nc+1):
                    vectors[index] += (ia-na) * cell[0]
                    vectors[index] += (ib-nb) * cell[1]
                    vectors[index] += (ic-nc) * cell[2]
                    
                    index += 1
                    
        self._supercell_vector_list = vectors

    #------- Methods : Oribital projeccted bandstructure -------#

    def orbital_projected_bandstructure(self, select):
        self._ensure_wfsx_hsx()
        # Select target orbitals
        if isinstance(select, str):
            select = [select]
        self.orbital_mask(select)
        self.mask_to_pointer()
        
        gamma = self.gamma
        
        # WFSX information
        wf = self.wavefunction
        kpt = self.kpoints
        eig = self.eigenvalue

        # Selected orbitals
        target = self._target
        list_io = self._projection_io
        list_ptr = self._projection_ptr

        # HSX information
        Sover = self.Sover
        xij = self.xij
        
        nwavefunctions = eig.shape[0]
        nspin = eig.shape[1]
        nkpoints = eig.shape[2]
        ntarget = len(target)
        
        fat = np.zeros((ntarget, nkpoints, nwavefunctions), dtype = float)
        
        for itar in range(ntarget):
            tar = target[itar]
            for ik in range(nkpoints): # loop over k-points
                print('Loop over kpoints : %d / %d'%(ik+1, nkpoints))
                for isp in range(nspin): # loop over spin
                    for iw in range(nwavefunctions): # loop over wavefunctions
                        nprojection = len(list_io[itar])
                        buff = np.zeros((nprojection), dtype = float)
                        list_target_io = tar['orbital_index']
                        for io1 in list_target_io: # list of target orbitals
                            iio = 0
                            for io2 in list_io[itar]: # list of projection orbitals
                                ind = list_ptr[itar][iio]
                                if (gamma==1):
                                    qcos = wf[0][io1]*wf[0][io2]
                                    qsin = 0
                                elif (gamma==0):
                                    qcos = wf[0][io1][iw][isp][ik] * wf[0][io2][iw][isp][ik] + \
                                           wf[1][io1][iw][isp][ik] * wf[1][io2][iw][isp][ik]
                                    qsin = wf[0][io1][iw][isp][ik] * wf[1][io2][iw][isp][ik] - \
                                           wf[1][io1][iw][isp][ik] * wf[0][io2][iw][isp][ik]
                                phase = (kpt[ik] * xij[ind]).sum() # phase fator
                                factor = qcos * np.cos(phase) - qsin * np.sin(phase)
                                buff[iio] = Sover[ind] * factor                            
                                iio += 1
                                
                        fat[itar][ik][iw] = buff.sum() # sum over projection orbitals
        self.fat = fat
        return fat

    #------- Methods : Oribital projeccted density of state -------#

    def orbital_projected_denstiy_of_state(self, select, energys):
        self._ensure_wfsx_hsx()
        # Select range of energy
        energys = np.array(energys)
        nenergy = len(energys)
        
        # Select target orbitals
        if isinstance(select, str):
            select = [select]
        self.orbital_mask(select)
        self.mask_to_pointer()
        
        gamma = self.gamma
        
        # WFSX information
        wf = self.wavefunction
        wk = self.kweight
        kpt = self.kpoints
        eig = self.eigenvalue

        # Selected orbitals
        target = self._target
        list_io = self._projection_io
        list_ptr = self._projection_ptr

        # HSX information
        Sover = self.Sover
        xij = self.xij
        
        nwavefunctions = eig.shape[0]
        nspin = eig.shape[1]
        nkpoints = eig.shape[2]
        ntarget = len(target)


        pdos = np.zeros((ntarget, nenergy), dtype = float)
        
        buff0 = np.zeros((nenergy, nkpoints), dtype = float)
        buff1 = np.zeros((nkpoints, nwavefunctions), dtype = float)
        
        for itar in range(ntarget):
            tar = target[itar]
            for ik in range(nkpoints): # loop over k-points
                print('Loop over kpoints : %d / %d'%(ik+1, nkpoints))
                for isp in range(nspin): # loop over spin
                    for iw in range(nwavefunctions): # loop over wavefunctions
                        nprojection = len(list_io[itar])
                        buff2 = np.zeros((nprojection), dtype = float)
                        list_target_io = tar['orbital_index']
                        for io1 in list_target_io: # list of target orbitals
                            iio = 0
                            for io2 in list_io[itar]: # list of projection orbitals
                                ind = list_ptr[itar][iio]
                                if (gamma==1):
                                    qcos = wf[0][io1]*wf[0][io2]
                                    qsin = 0
                                elif (gamma==0):
                                    qcos = wf[0][io1][iw][isp][ik] * wf[0][io2][iw][isp][ik] + \
                                           wf[1][io1][iw][isp][ik] * wf[1][io2][iw][isp][ik]
                                    qsin = wf[0][io1][iw][isp][ik] * wf[1][io2][iw][isp][ik] - \
                                           wf[1][io1][iw][isp][ik] * wf[0][io2][iw][isp][ik]
                                alfa = (kpt[ik] * xij[ind]).sum() # phase fator
                                factor = qcos * np.cos(alfa) - qsin * np.sin(alfa)
                                buff2[iio] = Sover[ind] * factor                            
                                iio += 1
                        buff1[itar][ik][iw] = buff2.sum() # sum over projection orbitals
                    
                        # wavefunction loop
                        eigenvalue = eig[iw][isp][ik] 
                        for ie in range(nenergy):
                            factor = self.delta(energys[ie]-eigenvalue)
                            buff0[ie][ik] = factor * buff1[ik].sum() * wk[ik]
            # first loop
            for ie in range(nenergy):        
                pdos[itar][ie] = buff0[ie].sum()
                       
        self.pdos = pdos
        return pdos

    #------- Methods : Oribital projeccted local density of states -------#

   # @profile
    def orbital_projected_local_density_of_state(self, select, energys, cell, mesh):
        self._ensure_wfsx_hsx()
        self._ensure_struct_supercell()
        # Select range of energy
        energys = np.array(energys)
        nenergy = len(energys)
    
        # Select target orbitals
        if isinstance(select, str):
            select = [select]
        self.orbital_mask(select)
        self.mask_to_pointer()
        
        # Define number of grid points from readGrid 양식 입력
        xgrid, ygrid, zgrid = self.unit_cell_grid(cell, mesh)
        
        # WFSX information
        gamma = self.gamma
        wf = self.wavefunction
        wk = self.kweight
        kpt = self.kpoints
        eig = self.eigenvalue

        index = self.atom_index
        symbol = self._atom_symbol
        print(index)
        
        # Selected orbitals
        target = self._target
        list_io = self._projection_io
        list_ptr = self._projection_ptr

        # HSX information
        Sover = self.Sover
        xij = self.xij
        
        # Struct information
        cell = self.cell
        atoms = self.atoms
        origin = atoms[0] # first atom
        supercell_vectors = self._supercell_vector_list
        nvectors = len(supercell_vectors)
        
        # Grid information
        na = int(mesh[0])
        nb = int(mesh[1])
        nc = int(mesh[2])

        # Parallelization ( Developing .. )

        nwavefunctions = eig.shape[0]
        nspin = eig.shape[1]
        nkpoints = eig.shape[2]
        ntarget = len(target)

        spatial = np.zeros((2, nvectors), dtype= float)
        pldos = np.zeros((ntarget, nenergy, na, nb, nc), dtype = float)
        
        for itar in range(ntarget):
            start = time.time()
            tar = target[itar]
            
            for ix in range(na):
                for iy in range(nb):
                    for iz in range(nc):
                        print('[ %d %d %d grid ]\n' %(ix+1,iy+1,iz+1))
                        print("time :", time.time() - start)
                        position_vector = np.zeros((3), dtype = float)
                        position_vector[0] = xgrid[0][ix][iy][iz]
                        position_vector[1] = ygrid[0][ix][iy][iz]
                        position_vector[2] = zgrid[0][ix][iy][iz]
                        list_target_io = tar['orbital_index']
                        number_of_target = len(list_target_io)
                        buff0 = np.zeros((nenergy, number_of_target), dtype = float)
                        iio1 = 0
                        for io1 in list_target_io: # list of target orbitals
                            #print('[%d / %d] target orbitals \n' %(iio1+1, number_of_target))
                            atom_symbol = symbol[io1]
                            atom_index = index[io1] - 1
                            target_position = atoms[atom_index]
                            target_vector = target_position - position_vector # grid point to target atom        
                            target_n = tar['principle_quantum_number'][iio1]
                            target_l = tar['angular_quantum_number'][iio1]
                            target_m = tar['magnetic_quantum_nuber'][iio1]
                            target_z = tar['zeta'][iio1]
                            
                            buff1 = np.zeros((nenergy, 2, nkpoints, nspin, nwavefunctions), dtype = float)
                            buff = np.zeros((nenergy, nkpoints), dtype = float)
                            for ik in range(nkpoints): # loop over k-points
                                #print('Loop over kpoints : %d / %d'%(ik+1, nkpoints))
                                for isp in range(nspin): # loop over spin
                                    for iw in range(nwavefunctions): # loop over wavefunctions
                                        eigenvalue = eig[iw][isp][ik]
                                        nprojection = len(list_io[itar])
                                        list_target_io = tar['orbital_index']
                                        buff2 = np.zeros((2, nprojection), dtype = float)
                                        iio2 = 0
                                        for io2 in list_io[itar]: # list of projection orbitals
                                            ind = list_ptr[itar][iio2]
                                            if (gamma==1):
                                                qcos = wf[0][io1]*wf[0][io2]
                                                qsin = 0
                                            elif (gamma==0):
                                                qcos = wf[0][io1][iw][isp][ik] * wf[0][io2][iw][isp][ik] + \
                                                    wf[1][io1][iw][isp][ik] * wf[1][io2][iw][isp][ik]
                                                qsin = wf[0][io1][iw][isp][ik] * wf[1][io2][iw][isp][ik] - \
                                                    wf[1][io1][iw][isp][ik] * wf[0][io2][iw][isp][ik]

                                            alfa = np.inner(kpt[ik], xij[ind]) # phase fator
                                            buff2[0][iio2] = qcos * math.cos(alfa) - qsin * math.sin(alfa)
                                            buff2[1][iio2] = qcos * math.sin(alfa) + qsin * math.sin(alfa)
                                            buff2 *=  Sover[ind]
                                            iio2 += 1
                                                                                    
                                        rval = (buff2[0]).sum()
                                        ival = (buff2[1]).sum()
                                        for ie in range(nenergy):
                                            factor = self.delta(energys[ie]-eigenvalue)
                                            buff1[ie][0][ik][isp][iw] = rval * factor
                                            buff1[ie][1][ik][isp][iw] = ival * factor
                                
                                # k loop

                                buff1 = wk[ik] * buff1

                                for iv in range(nvectors):
                                    xji = -(target_vector + supercell_vectors[iv])
                                    phase = np.inner(kpt[ik], xji)
                                    r = np.sqrt(xji.dot(xji))
                                    phir = self.Rnl(atom_symbol, target_n, target_l, target_z, r)
                                    
                                    if phir < phi_tolerance:
                                        factor = 0
                                    else:
                                       spherical = self.Yml(xji, target_m, target_l) # imaginary
                                       factor = phir * spherical 
                                        
                                    spatial[0][iv] = factor.real * math.cos(phase) - factor.imag * math.sin(phase)
                                    spatial[1][iv] = factor.real * math.sin(alfa) + factor.imag * math.cos(phase)
                                rspatial = spatial[0].sum()
                                ispatial = spatial[1].sum()
                                
                                for ie in range(nenergy):
                                    rphi_orbital = buff1[ie][0][ik].sum()
                                    iphi_orbital = buff1[ie][1][ik].sum()
                                    buff[ie][ik] = rphi_orbital * rspatial - iphi_orbital * ispatial

                            # io1 loop        
                            for ie in range(nenergy):
                                buff0[ie][iio1] = buff[ie].sum()
                            iio1 += 1
                            
                        # xyz loop
                        for ie in range(nenergy):
                            pldos[itar][ie][ix][iy][iz] = buff0[ie].sum()
                        

        self.pldos = pldos
        return pldos
        
