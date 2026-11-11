def compute_rb_to_h_transform():
        model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        data = mujoco.MjData(model)
        
        #R(RB←H) = R(RB←World) · R(World←H)
        #Obtaining each R(RB←World)
        T_RB_World = {}
        for side, body_name in PORT_BODIES.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"Body not found: {body_name}")
            T = np.eye(4)
            T[:3, :3] = data.xmat
            T[:3, 3] = data.xpos[body_id]
            T_RB_World[side] = np.linalg.inv(T)

        #Obtaining  R(World←H) (can be mocaps position, but mocap cannot be attatched to the tool anymore, it has to be only the virtual representation of the TCP inside MJC)
        T_World_H = {}
        for side, body_name in PORT_BODIES.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"Body not found: {body_name}")
            
            pos = data.xpos[body_id]
            mat = data.xmat[body_id].reshape(3, 3)

            mat_4x4 = np.eye(4)
            mat_4x4[:3, :3] = mat
            mat_4x4[:3, 3] = pos 
            T_World_H[side] = mat_4x4

        #Final T_RB_H
        T_RB_H = {}
        for side in PORT_BODIES.keys():
             T_RB_H[side] = T_RB_World[side]@T_World_H[side]
             
        return T_RB_H