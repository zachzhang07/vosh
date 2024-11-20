import numpy as np
import pymeshlab as pml


# import pymeshfix


def isotropic_explicit_remeshing(verts, faces):
    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')  # will copy!

    # filters
    # ms.apply_coord_taubin_smoothing()
    ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] isotropic explicit remesh: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


def generate_from_selected(verts, faces, verts_global, faces_global, mask):
    # _ori_vert_shape = verts.shape
    # _ori_face_shape = faces.shape

    # m = pml.Mesh(verts, faces)
    m_global = pml.Mesh(verts_global, faces_global, f_scalar_array=mask)  # mask as the quality
    ms = pml.MeshSet()
    # ms.add_mesh(m, 'mesh')
    ms.add_mesh(m_global, 'mesh_global')

    # select faces
    ms.compute_selection_by_condition_per_face(condselect='fq == 1')  # select
    ms.generate_from_selected_faces()  # now the current mesh is the new mesh

    ms.set_current_mesh(0)
    m = ms.current_mesh()
    verts_global = m.vertex_matrix()
    faces_global = m.face_matrix()
    ms.set_current_mesh_visibility(False)

    if verts is not None:
        m = pml.Mesh(verts, faces)
        ms.add_mesh(m, 'mesh')

    ms.generate_by_merging_visible_meshes()
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    return verts, faces, verts_global, faces_global


def decimate_mesh(verts, faces, target, backend='pymeshlab', remesh=False, optimalplacement=True):
    # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == 'pyfqmr':
        import pyfqmr
        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=int(target), preserve_border=False, verbose=False)
        verts, faces, normals = solver.getMesh()
    else:

        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, 'mesh')  # will copy!

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.Percentage(1))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target), optimalplacement=optimalplacement)

        if remesh:
            ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

        # extract mesh
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    print(f'[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


def remove_masked_trigs(verts, faces, mask, dilation=5):
    # mask: 0 == keep, 1 == remove

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces, f_scalar_array=mask)  # mask as the quality
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')  # will copy!

    # select faces
    ms.compute_selection_by_condition_per_face(condselect='fq == 0')  # select kept faces
    # dilate to aviod holes...
    for _ in range(dilation):
        ms.apply_selection_dilatation()
    ms.apply_selection_inverse(invfaces=True)  # invert

    # delete faces
    ms.meshing_remove_selected_faces()

    # clean unref verts
    ms.meshing_remove_unreferenced_vertices()

    # extract
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh mask trigs: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


def remove_masked_verts(verts, faces, mask):
    # mask: 0 == keep, 1 == remove

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces, v_scalar_array=mask)  # mask as the quality
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')  # will copy!

    # select verts
    ms.compute_selection_by_condition_per_vertex(condselect='q == 1')

    # delete verts and connected faces
    ms.meshing_remove_selected_vertices()

    # extract
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh mask verts: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


def remove_selected_verts(verts, faces, query='(x < 1) && (x > -1) && (y < 1) && (y > -1) && (z < 1 ) && (z > -1)'):
    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')  # will copy!

    # select verts
    ms.compute_selection_by_condition_per_vertex(condselect=query)

    # delete verts and connected faces
    ms.meshing_remove_selected_vertices()

    # extract
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh remove verts: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


def remove_selected_vt_by_edge_length(verts, faces, threshold=1.0):
    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')  # will copy!

    # select verts
    ms.compute_selection_by_edge_length(threshold=threshold)

    # delete verts and connected faces
    ms.meshing_remove_selected_vertices_and_faces()

    # extract
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh remove verts by edge length: {_ori_vert_shape} --> {verts.shape}, '
          f'{_ori_face_shape} --> {faces.shape}')

    return verts, faces


def remove_selected_isolated_faces(verts, faces, mincomponentsize=100):
    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')  # will copy!

    # delete verts and connected faces
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize=mincomponentsize)

    # extract
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh remove selected isolated verts and faces: {_ori_vert_shape} --> {verts.shape}, '
          f'{_ori_face_shape} --> {faces.shape}')

    return verts, faces


def close_holes_meshlab(verts, faces, maxholesize=30):
    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')  # will copy!

    # repair
    ms.set_selection_none(allfaces=True)
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    # delete verts and connected faces
    ms.meshing_close_holes(maxholesize=maxholesize)

    # extract
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh close holes smaller than a given threshold {maxholesize}: {_ori_vert_shape} --> {verts.shape}, '
          f'{_ori_face_shape} --> {faces.shape}')

    return verts, faces


# def close_holes_meshfix(verts, faces, nbe=30):
#     _ori_vert_shape = verts.shape
#     _ori_face_shape = faces.shape
#
#     # Create TMesh object
#     tin = pymeshfix.PyTMesh()
#
#     tin.load_array(verts, faces)  # or read arrays from memory
#
#     # Fill holes
#     tin.fill_small_boundaries(nbe=nbe)
#     # print('There are {:d} boundaries'.format(tin.boundaries()))
#
#     # Clean (removes self-intersections)
#     # tin.clean(max_iters=10, inner_loops=3)
#
#     # Check mesh for holes again
#     # print('There are {:d} boundaries'.format(tin.boundaries()))
#
#     # or return numpy arrays
#     verts, faces = tin.return_arrays()
#
#     print(f'[INFO] mesh close holes smaller than a given threshold {nbe}: {_ori_vert_shape} --> {verts.shape}, '
#           f'{_ori_face_shape} --> {faces.shape}')
#
#     return verts, faces


def select_sharp_and_flat_faces_by_normal(verts, faces, usear=False, aratio=0.02, nfratio_sharp=120, nfratio_flat=5):
    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')

    sharp_faces_mask, flat_faces_mask = None, None

    if nfratio_sharp > 0:
        ms.compute_selection_bad_faces(usear=usear, aratio=aratio, usenf=True, nfratio=nfratio_sharp)
        m = ms.current_mesh()
        sharp_faces_mask = m.face_selection_array()
        ms.set_selection_none(allfaces=True)

    if nfratio_flat > 0:
        ms.compute_selection_bad_faces(usear=usear, aratio=aratio, usenf=True, nfratio=nfratio_flat)
        m = ms.current_mesh()
        flat_faces_mask = m.face_selection_array() == False  # reverse
        ms.set_selection_none(allfaces=True)

    return sharp_faces_mask, flat_faces_mask


def select_sharp_and_flat_faces_by_normal_using_ratio(verts, faces, sharp_ratio=0.05, flat_ratio=0.1):
    assert 1.0 > sharp_ratio >= 0 and 1.0 > flat_ratio >= 0
    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')

    sharp_faces_mask, flat_faces_mask = None, None

    nfratio_sharp = 175
    while sharp_ratio > 0 and nfratio_sharp > 0 and sharp_faces_mask is None:
        ms.compute_selection_bad_faces(usear=False, usenf=True, nfratio=nfratio_sharp)
        m = ms.current_mesh()
        sharp_faces_mask = m.face_selection_array()
        if sharp_faces_mask.sum() / len(sharp_faces_mask) < sharp_ratio:
            nfratio_sharp -= 10
            sharp_faces_mask = None
        ms.set_selection_none(allfaces=True)
        # print('nfratio_sharp: ', nfratio_sharp)
        # if sharp_faces_mask is not None:
        #     print('sharp_faces_mask.sum() / len(sharp_faces_mask): ', sharp_faces_mask.sum() / len(sharp_faces_mask))

    nfratio_flat = 5
    while flat_ratio > 0 and nfratio_flat < 180 and flat_faces_mask is None:
        ms.compute_selection_bad_faces(usear=False,usenf=True, nfratio=nfratio_flat)
        m = ms.current_mesh()
        flat_faces_mask = m.face_selection_array() == False  # reverse
        if flat_faces_mask.sum() / len(flat_faces_mask) < flat_ratio:
            nfratio_flat += 10
            flat_faces_mask = None
        ms.set_selection_none(allfaces=True)
        # print('nfratio_flat: ', nfratio_flat)
        # if flat_faces_mask is not None:
        #     print('flat_faces_mask.sum() / len(flat_faces_mask): ', flat_faces_mask.sum() / len(flat_faces_mask))


    return sharp_faces_mask, flat_faces_mask


def clean_mesh(verts, faces, v_pct=1, min_f=8, min_d=5, repair=True, remesh=True):
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')  # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces

    if v_pct > 0:
        ms.meshing_merge_close_vertices(threshold=pml.Percentage(v_pct))  # 1/10000 of bounding box diagonal

    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pml.Percentage(min_d))

    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


def decimate_and_refine_mesh(verts, faces, mask, decimate_ratio=0.1, refine_size=0.01, refine_remesh_size=0.02):
    # verts: [N, 3]
    # faces: [M, 3]
    # mask: [M], 0 denotes do nothing, 1 denotes decimation, 2 denotes subdivision

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces, f_scalar_array=mask)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')  # will copy!

    # repair
    ms.set_selection_none(allfaces=True)
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    # decimate and remesh
    ms.compute_selection_by_condition_per_face(condselect='fq == 1')
    if decimate_ratio > 0:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int((1 - decimate_ratio) * (mask == 1).sum()),
                                                    selected=True)

    if refine_remesh_size > 0:
        ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.AbsoluteValue(refine_remesh_size),
                                                selectedonly=True)

    # repair
    ms.set_selection_none(allfaces=True)
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    # refine 
    if refine_size > 0:
        ms.compute_selection_by_condition_per_face(condselect='fq == 2')
        ms.meshing_surface_subdivision_midpoint(threshold=pml.AbsoluteValue(refine_size), selected=True)

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh decimating & subdividing: {_ori_vert_shape} --> {verts.shape}, '
          f'{_ori_face_shape} --> {faces.shape}')

    return verts, faces

# in meshutils.py
def select_bad_and_flat_faces_by_normal(verts, faces, usear=False, aratio=0.02, nfratio_bad=120, nfratio_flat=20):
    m = pml.Mesh(verts, faces)

    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')

    ms.compute_selection_bad_faces(usear=usear, aratio=aratio, usenf=True, nfratio=nfratio_bad)
    bad_faces_mask = ms.current_mesh().face_selection_array()
    ms.set_selection_none(allfaces=True)

    ms.compute_selection_bad_faces(usear=usear, aratio=aratio, usenf=True, nfratio=nfratio_flat)
    flat_faces_mask = ms.current_mesh().face_selection_array() == False  # reverse
    ms.set_selection_none(allfaces=True)

    return bad_faces_mask, flat_faces_mask
