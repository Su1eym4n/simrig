"""Serialize compiled MuJoCo geometry for browser-side Three.js rendering."""

from __future__ import annotations

from typing import Any

import numpy as np


def scene_payload(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    model_name: str,
) -> dict[str, Any]:
    """Return visual geometry and current world transforms for a model."""

    visible_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_group[geom_id]) <= 2
    ]
    used_mesh_ids = sorted(
        {
            int(model.geom_dataid[geom_id])
            for geom_id in visible_geom_ids
            if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
            and int(model.geom_dataid[geom_id]) >= 0
        }
    )
    return {
        "model_name": model_name,
        "coordinate_system": "z-up",
        "meshes": [_mesh_payload(mujoco, model, mesh_id) for mesh_id in used_mesh_ids],
        "geoms": [_geom_payload(mujoco, model, geom_id) for geom_id in visible_geom_ids],
        "transforms": geom_transforms(model, data),
    }


def geom_transforms(model: Any, data: Any) -> list[dict[str, Any]]:
    """Return each geom's current MuJoCo world position and rotation matrix."""

    return [
        {
            "id": geom_id,
            "position": np.asarray(data.geom_xpos[geom_id], dtype=float).tolist(),
            "matrix": np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(-1).tolist(),
        }
        for geom_id in range(model.ngeom)
    ]


def _mesh_payload(mujoco: Any, model: Any, mesh_id: int) -> dict[str, Any]:
    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    face_adr = int(model.mesh_faceadr[mesh_id])
    face_num = int(model.mesh_facenum[mesh_id])
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
    return {
        "id": mesh_id,
        "name": name or f"mesh_{mesh_id}",
        "vertices": np.asarray(
            model.mesh_vert[vert_adr : vert_adr + vert_num],
            dtype=np.float32,
        )
        .reshape(-1)
        .tolist(),
        "indices": np.asarray(
            model.mesh_face[face_adr : face_adr + face_num],
            dtype=np.uint32,
        )
        .reshape(-1)
        .tolist(),
    }


def _geom_payload(mujoco: Any, model: Any, geom_id: int) -> dict[str, Any]:
    material_id = int(model.geom_matid[geom_id])
    if material_id >= 0:
        rgba = np.asarray(model.mat_rgba[material_id], dtype=float)
        specular = float(model.mat_specular[material_id])
        shininess = float(model.mat_shininess[material_id])
        reflectance = float(model.mat_reflectance[material_id])
        emission = float(model.mat_emission[material_id])
    else:
        rgba = np.asarray(model.geom_rgba[geom_id], dtype=float)
        specular = 0.25
        shininess = 0.25
        reflectance = 0.0
        emission = 0.0
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return {
        "id": geom_id,
        "name": name or f"geom_{geom_id}",
        "type": int(model.geom_type[geom_id]),
        "mesh_id": int(model.geom_dataid[geom_id]),
        "group": int(model.geom_group[geom_id]),
        "size": np.asarray(model.geom_size[geom_id], dtype=float).tolist(),
        "rgba": rgba.tolist(),
        "material": {
            "specular": specular,
            "shininess": shininess,
            "reflectance": reflectance,
            "emission": emission,
        },
    }
