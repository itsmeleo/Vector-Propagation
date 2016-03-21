
from scipy.spatial import kdtree
import os
import xml.etree.ElementTree as ET
from copy import copy
from scipy import spatial
import numpy as np
import libtiff
from scipy import ndimage as nd
from matplotlib import pyplot as plt
import networkx as nx
import community
from multiprocessing import Pool

from Function_for_multiproc import single_cell_propagation

class LineageTree(object):
    """docstring for LineageTree"""


    def _dist_v(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.sum((v1-v2)**2)**(.5)

    def copy_cell(self, C, links=[]):
        C_tmp = copy(C)
        self.nodes.append(C)

    def get_next_id(self):
        if self.next_id == []:
            self.max_id += 1
            return self.max_id
        else:
            return self.next_id.pop()

    def add_node(self, t, succ, pos):
        C_next = self.get_next_id()
        self.time_nodes[t].append(C_next)
        self.successor.setdefault(succ, []).append(C_next)
        self.predecessor.setdefault(C_next, []).append(succ)
        self.edges.append((succ, C_next))
        self.nodes.append(C_next)
        self.pos[C_next] = pos
        self.progeny[C_next] = 0
        return C_next

    def remove_node(self, c):
        self.nodes.remove(c)
        for t, Cs in self.time_nodes.iteritems():
            if c in Cs:
                Cs.remove(c)
        # self.time_nodes.pop(c, 0)
        pos = self.pos.pop(c, 0)
        e_to_remove = [e for e in self.edges if c in e]
        for e in e_to_remove:
            self.edges.remove(e)
        if c in self.roots:
            self.roots.remove(c)
        succ = self.successor.pop(c, [])
        s_to_remove = [s for s, ci in self.successor.iteritems() if c in ci]
        for s in s_to_remove:
            self.successor[s].remove(c)

        pred = self.predecessor.pop(c, [])
        p_to_remove = [s for s, ci in self.predecessor.iteritems() if ci == c]
        for s in p_to_remove:
            self.predecessor[s].remove(c)

        self.time.pop(c, 0)
        self.spatial_density.pop(c, 0)

        self.next_id.append(c)
        return e_to_remove, succ, s_to_remove, pred, p_to_remove, pos

    def fuse_nodes(self, c1, c2):
        e_to_remove, succ, s_to_remove, pred, p_to_remove, c2_pos = self.remove_node(c2)
        for e in e_to_remove:
            new_e = [c1] + [other_c for other_c in e if e != c2]
            self.edges.append(new_e)

        self.successor.setdefault(c1, []).extend(succ)
        self.predecessor.setdefault(c1, []).extend(pred)

        for s in s_to_remove:
            self.successor[s].append(c1)

        for p in p_to_remove:
            self.predecessor[p].append(c1)

        self.pos[c1] = np.mean([self.pos[c1], c2_pos], axis = 0)
        self.progeny[c1] += 1


    def to_tlp(self, fname, t_min=-1, t_max=np.inf, temporal=True, spatial=False, VF=False):
        """
        Write a lineage tree into an understable tulip file
        fname : path to the tulip file to create
        lin_tree : lineage tree to write
        properties : dictionary of properties { 'Property name': [{c_id: prop_val}, default_val]}
        """
        
        f=open(fname, "w")

        f.write("(tlp \"2.0\"\n")
        f.write("(nodes ")
        if t_max!=np.inf or t_min>-1:
            nodes_to_use = [n for n in self.nodes if t_min<n.time<=t_max]
            edges_to_use = []
            if temporal:
                edges_to_use += [e for e in self.edges if t_min<e[0].time<t_max]
            if spatial:
                edges_to_use += [e for e in self.spatial_edges if t_min<e[0].time<t_max]
        else:
            nodes_to_use = self.nodes
            edges_to_use = []
            if temporal:
                edges_to_use += self.edges
            if spatial:
                edges_to_use += self.spatial_edges

        for n in nodes_to_use:
            f.write(str(n)+ " ")
        f.write(")\n")

        for i, e in enumerate(edges_to_use):
            f.write("(edge " + str(i) + " " + str(e[0]) + " " + str(e[1]) + ")\n")
        # f.write("(property 0 int \"id\"\n")
        # f.write("\t(default \"0\" \"0\")\n")
        # for n in nodes_to_use:
        #     f.write("\t(node " + str(n) + str(" \"") + str(self.n) + "\")\n")
        # f.write(")\n")

        f.write("(property 0 int \"time\"\n")
        f.write("\t(default \"0\" \"0\")\n")
        for n in nodes_to_use:
            f.write("\t(node " + str(n) + str(" \"") + str(self.time[n]) + "\")\n")
        f.write(")\n")

        f.write("(property 0 layout \"viewLayout\"\n")
        f.write("\t(default \"(0, 0, 0)\" \"()\")\n")
        for n in nodes_to_use:
            f.write("\t(node " + str(n) + str(" \"") + str(tuple(self.pos[n])) + "\")\n")
        f.write(")\n")

        f.write("(property 0 double \"distance\"\n")
        f.write("\t(default \"0\" \"0\")\n")
        for i, e in enumerate(edges_to_use):
            d_tmp = self._dist_v(self.pos[e[0]], self.pos[e[1]])
            f.write("\t(edge " + str(i) + str(" \"") + str(d_tmp) + "\")\n")
            f.write("\t(node " + str(e[0]) + str(" \"") + str(d_tmp) + "\")\n")
        f.write(")\n")

        # for property in properties:
        #     prop_name=property[0]
        #     vals=property[1]
        #     default=property[2]
        #     f.write("(property 0 string \""+prop_name+"\"\n")
        #     f.write("\t(default \""+str(default)+"\" \"0\")\n")
        #     for node in nodes:
        #         f.write("\t(node " + str(node) + str(" \"") + str(vals.get(node, default)) + "\")\n")
        #     f.write(")\n") 
        f.write(")")
        f.close()

    def median_average(self, subset):
        subset_dist = [np.mean([di.pos for di in c.D], axis = 0) - c.pos for c in subset if c.D != []]
        target_C = [c for c in subset if c.D != []]
        if subset_dist != []:
            med_distance = spatial.distance.squareform(spatial.distance.pdist(subset_dist))
            return subset_dist[np.argmin(np.sum(med_distance, axis=0))]
        else:
            return [0, 0, 0]

    def median_average_bw(self, subset):
        subset_dist = [c.M.pos - c.pos for c in subset if c.M != self.R]
        target_C = [c for c in subset if c.D != []]
        if subset_dist != []:
            med_distance = spatial.distance.squareform(spatial.distance.pdist(subset_dist))
            return subset_dist[np.argmin(np.sum(med_distance, axis=0))]
        else:
            return [0, 0, 0]

    def build_median_vector(self, C, dist_th, delta_t = 2):#temporal_space=lambda d, t, c: d+(t*c)):
        if not hasattr(self, 'spatial_edges'):
            self.compute_spatial_edges(dist_th)
        subset = [C]
        subset += C.N
        added_D = added_M = subset
        for i in xrange(delta_t):
            _added_D = []
            _added_M = []
            for c in added_D:
                _added_D += c.D
            for c in added_M:
                if not c.M is None:
                    _added_M += [c.M]
            subset += _added_M
            subset += _added_D
            added_D = _added_D
            added_M = _added_M


        return self.median_average(subset)

    def build_vector_field(self, dist_th=50):
        ruler = 0
        for C in self.nodes:
            if ruler != C.time:
                print C.time
            C.direction = self.build_median_vector(C, dist_th)
            ruler = C.time
    
    def single_cell_propagation(self, params):
        C, t, nb_max, dist_max, to_check_self, R, pos, successor, predecessor = params
        idx3d = self.kdtrees[t]
        closest_cells = np.array(to_check_self)[list(idx3d.query(tuple(pos[C]), nb_max)[1])]
        max_value = np.min(np.where(np.array([_dist_v(pos[C], pos[ci]) for ci in closest_cells]+[dist_max+1])>dist_max))
        cells_to_keep = closest_cells[:max_value]
        # med = median_average_bw(cells_to_keep, R, pos)
        # print type (cells_to_keep)
        subset_dist = [np.mean([pos[cii] for cii in predecessor[ci]], axis=0) - pos[ci] for ci in cells_to_keep if not ci in R]
        if subset_dist != []:
            med_distance = spatial.distance.squareform(spatial.distance.pdist(subset_dist))
            med = subset_dist[np.argmin(np.sum(med_distance, axis=0))]
        else:
            med = [0, 0, 0]
        return C, med
    
    def read_from_xml(self, file_format, tb, te, z_mult=1.):
        self.time_nodes = {}
        self.time_edges = {}
        unique_id = 0
        self.nodes = []
        self.edges = []
        self.roots = []
        self.successor = {}
        self.predecessor = {}
        self.pos = {}
        self.time_id = {}
        self.time = {}
        for t in range(tb, te+1):
            t_str = '%04d' % t
            tree = ET.parse(file_format.replace('$TIME$', t_str))
            root = tree.getroot()
            self.time_nodes[t] = []
            self.time_edges[t] = []
            for it in root.getchildren():
                M_id, pos, cell_id = (int(it.attrib['parent']), 
                                      [float(v) for v in it.attrib['m'].split(' ') if v!=''], 
                                      int(it.attrib['id']))
                C = unique_id
                pos[-1] = pos[-1]*z_mult
                if self.time_id.has_key((t-1, M_id)):
                    M = self.time_id[(t-1, M_id)]
                    self.successor.setdefault(M, []).append(C)
                    self.predecessor.setdefault(C, []).append(M)
                    self.edges.append((M, C))
                    self.time_edges[t].append((M, C))
                else:
                    self.roots.append(C)
                self.pos[C] = pos
                self.nodes.append(C)
                self.time_nodes[t].append(C)
                self.time_id[(t, cell_id)] = C
                self.time[C] = t
                unique_id += 1

        self.max_id = unique_id - 1

    def get_idx3d(self, t):
        to_check_self = self.time_nodes[t]
        if not self.kdtrees.has_key(t):
            data_corres = {}
            data = []
            for i, C in enumerate(to_check_self):
                data.append(tuple(self.pos[C]))
                data_corres[i] = C
            idx3d = kdtree.KDTree(data)
            self.kdtrees[t] = idx3d
        else:
            idx3d = self.kdtrees[t]
        return idx3d, to_check_self

    def build_VF_propagation_backward(self, t_b=0, t_e=200, nb_max=20, dist_max=200, nb_proc = 8):
        VF = LineageTree(None, None, None)

        # Hack to allow pickling of kdtrees for multiprocessing
        kdtree.node = kdtree.KDTree.node
        kdtree.leafnode = kdtree.KDTree.leafnode
        kdtree.innernode = kdtree.KDTree.innernode
        if self.spatial_density == {}:
            self.compute_spatial_density(t_e, t_b, nb_max)

        starting_cells = self.time_nodes[t_b]
        unique_id = 0
        VF.time_nodes = {t_b: []}
        for C in starting_cells:
            # C_tmp = CellSS(unique_id=unique_id, id=unique_id, M=VF.R, time = t_b, pos = C.pos)
            i = VF.get_next_id()
            VF.nodes.append(i)
            VF.time_nodes[t_b].append(i)
            VF.roots.append(i)
            VF.pos[i]=self.pos[C]

        for t in range(t_b, t_e, -1):
            print t,
            to_check_VF = VF.time_nodes[t]

            idx3d, to_check_self = self.get_idx3d(t)

            VF.time_nodes[t-1] = []
            mapping = []
            tmp = np.array(to_check_self)
            for C in to_check_VF:
                mapping += [(C, idx3d, nb_max, dist_max, tmp, self.roots, self.pos, VF.pos, self.successor, self.predecessor)]
            
            out = []

            if nb_proc<2:
                for params in mapping:
                  out += [single_cell_propagation(params)]
            else:
                pool = Pool(processes=nb_proc)
                out = pool.map(single_cell_propagation, mapping)
                pool.terminate()
                pool.close()
            for C, med in out:
                VF.add_node(t-1, C, VF.pos[C] + med)
                # C_next = VF.get_next_id()
                # # C_next = CellSS(unique_id, unique_id, M=C, time = t-1, pos= C.pos + med)
                # VF.time_nodes[t-1].append(C_next)
                # VF.successor.setdefault(C, []).append(C_next)
                # VF.edges.append((C, C_next))
                # VF.nodes.append(C_next)
                # VF.pos[C_next] = VF.pos[C] + med

            idx3d, to_check_self = self.get_idx3d(t-1)
            to_check_VF = VF.time_nodes[t-1]

            equivalence = idx3d.query([VF.pos[c] for c in to_check_VF], 2)[1]
            equivalence = equivalence[:,1]

            count = np.bincount(equivalence)
            LT_too_close, = np.where(count > 1)
            TMP = []
            for C_LT in LT_too_close:
                to_potentially_fuse, = np.where(equivalence == C_LT)
                pos_tmp = [VF.pos[c] for c in to_potentially_fuse]
                dist_tmp = spatial.distance.squareform(spatial.distance.pdist(pos_tmp))
                dist_tmp[dist_tmp==0] = np.inf
                if (dist_tmp<self.spatial_density[to_check_self[C_LT]]/2.).any():
                    to_fuse = np.where(dist_tmp == np.min(dist_tmp))[0]
                    c1, c2 = to_potentially_fuse[list(to_fuse)]
                    to_check_VF[c1], to_check_VF[c2]
                    TMP.append([to_check_VF[c1], to_check_VF[c2]])

            for c1, c2 in TMP:
                # new_pos = np.mean([VF.pos[c1], VF.pos[c2]], axis = 0)
                VF.fuse_nodes(c1, c2)


        VF.t_b = t_b
        VF.t_e = t_e
        print
        return VF

    def compute_spatial_density(self, t_b=0, t_e=200, n_size=10):
        for t in range(t_b, t_e+1, 1):
            Cs = self.time_nodes[t]
            if not self.kdtrees.has_key(t):
                data_corres = {}
                data = []
                for i, C in enumerate(Cs):
                    data.append(tuple(self.pos[C]))
                    data_corres[i] = C
                idx3d = kdtree.KDTree(data)
                self.kdtrees[t] = idx3d
            else:
                idx3d = self.kdtrees[t]
            distances, indices = idx3d.query(data, n_size)
            self.spatial_density.update(dict(zip(Cs, np.mean(distances[:, 1:], axis=1))))

    def compute_spatial_edges(self, th=50):
        self.spatial_edges=[]
        for t, Cs in self.time_nodes.iteritems():
            nodes_tmp, pos_tmp = zip(*[(C, C.pos) for C in Cs])
            nodes_tmp = np.array(nodes_tmp)
            distances = spatial.distance.squareform(spatial.distance.pdist(pos_tmp))
            nodes_to_match = np.where((0<distances) & (distances<th))
            to_link = zip(nodes_tmp[nodes_to_match[0]], nodes_tmp[nodes_to_match[1]])
            self.spatial_edges.extend(to_link)
            for C1, C2 in to_link:
                C1.N.append(C2)

    def __init__(self, file_format, tb, te, z_mult = .1):
        super(LineageTree, self).__init__()
        self.time_nodes = {}
        self.time_edges = {}
        self.max_id = -1
        self.next_id = []
        self.nodes = []
        self.edges = []
        self.roots = []
        self.successor = {}
        self.predecessor = {}
        self.pos = {}
        self.time_id = {}
        self.time = {}
        self.kdtrees = {}
        self.spatial_density = {}
        self.progeny = {}
        if not (file_format is None or tb is None or te is None):
            self.read_from_xml(file_format, tb, te, z_mult=z_mult)
            self.t_b = tb
            self.t_e = te

path_to_files = '/Users/guignardl/dvlp/DATA/GMEMtracking3D_2015_3_6_17_42_12_Drosophila_12_08_28_trainCDWT_iter5/XML_finalResult_lht'

LT = LineageTree(file_format = path_to_files + '/GMEMfinalResult_frame$TIME$.xml', tb = 0, te = 300, z_mult = 5.)

from time import time
t_beging_p = time()
VF = LT.build_VF_propagation_backward(t_b = 10, t_e = 7, nb_max = 10, nb_proc=8)
t_end_p = time()
print 'parallel processing:',  t_end_p - t_beging_p
