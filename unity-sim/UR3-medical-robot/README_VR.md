# VR Integration Guide

This document describes how to run the VR version of the dual-tool surgical simulation in Unity.

## 1. Required Branch

The VR implementation is currently maintained in:

```bash
git checkout vr-integration
git pull origin vr-integration
```

The standard non-VR scene remains available separately.

---

## 2. Unity Project

Open the following Unity project:

```text
unity-sim/UR3-medical-robot
```

Use the Unity version specified by the project.

After opening the project, allow Unity to restore all required packages automatically.

The VR integration includes:

- XR Plug-in Management
- OpenXR Plugin
- Meta OpenXR
- XR Interaction Toolkit
- Input System
- HMD calibration support

---

## 3. VR Scene

Open:

```text
Assets/Scenes/DualToolSceneVR.unity
```

Do not use `DualToolScene.unity` when testing VR.

The base scene is preserved for non-VR operation.

The VR hierarchy contains:

```text
HMD_Calibration_Target

HMD_Support_Anchor
└── XR_Origin
    └── Endoscope Camera
```

`Endoscope Camera` receives the pose of the physical HMD.

`HMD_Support_Anchor` provides the transformation used to align the physical HMD with the desired virtual endoscope position.

`HMD_Calibration_Target` defines the desired neutral camera pose in the surgical simulation.

---

## 4. HMD Calibration

The VR scene includes:

```text
Assets/Scripts/VR/HMDCalibration.cs
```

The `HMDCalibration` component is attached to:

```text
HMD_Support_Anchor
```

Its references should be:

```text
Tracked Camera:
Endoscope Camera

Calibration Target:
HMD_Calibration_Target
```

The default calibration key is:

```text
C
```

### Calibration procedure

1. Start the VR runtime and connect the headset.
2. Open `DualToolSceneVR`.
3. Press Play in Unity.
4. Place the HMD in a comfortable neutral position.
5. Give focus to the Unity Game window.
6. Press `C` on the computer keyboard.
7. The VR reference frame will be adjusted so that the current physical HMD pose matches `HMD_Calibration_Target`.

The calibration modifies `HMD_Support_Anchor`; it does not overwrite the tracked pose of `Endoscope Camera`.

The calibration can also be triggered from the `HMDCalibration` component using:

```text
Calibrate HMD
```

from its context menu.

---

## 5. Calibration Target

The initial target corresponds to the previously validated endoscope camera pose:

```text
Position:
X = 0
Y = 0.204637
Z = -0.09

Rotation:
X = 64.654
Y = 0
Z = 0
```

If a different working viewpoint is required, adjust `HMD_Calibration_Target` rather than manually modifying the tracked `Endoscope Camera`.

---

## 6. XR Device Simulator

The XR Device Simulator may be available for development without physical VR hardware.

For physical HMD testing, keep the simulator disabled to avoid conflicts with the real XR device.

The simulator was useful for validating rotational tracking, but positional translation through the simulator did not behave consistently during development.

Physical HMD tracking should therefore be used for final validation.

---

## 7. Rift CV1 / Meta PC Setup

For Oculus Rift CV1 testing:

1. Install and open Meta Horizon Link.
2. Configure the Rift and its external sensors.
3. Confirm that the headset works correctly in a native VR application before testing Unity.
4. Allow unknown sources if required by the Meta PC software.
5. Start Unity only after the Meta/Oculus runtime is running.

The Rift CV1 used during development successfully provided position and rotation tracking once the GPU configuration described below was corrected.

---

## 8. Hybrid GPU Troubleshooting

On laptops with both an integrated GPU and a dedicated GPU, the Oculus runtime and the VR application must use compatible graphics adapters.

A tested development laptop contained:

```text
Integrated GPU:
AMD Radeon(TM) Graphics

Dedicated GPU:
NVIDIA GeForce RTX 3050 Ti
```

Initially, Google Earth VR and Unity failed because the Oculus runtime and the VR application were using different graphics adapters.

The runtime reported an error similar to:

```text
Graphics adapters do not match between client and service
```

On this machine, the Rift worked when the relevant applications were assigned to the AMD GPU.

The following executables were configured to use the same AMD graphics adapter:

```text
OculusClient.exe
OVRServer_x64.exe
Unity.exe
Google Earth VR / Earth.exe
```

This configuration is machine-specific.

Do not automatically force every development computer to use AMD or the integrated GPU. Instead, make sure that:

```text
VR runtime
+
Unity
+
VR application
```

use the graphics adapter associated with the headset/display path.

On Windows, per-application GPU selection is available under:

```text
Settings
→ System
→ Display
→ Graphics
```

If Unity detects the headset but the scene does not appear in the HMD, or tracking remains at zero, verify the GPU configuration before changing the Unity XR hierarchy.

---

## 9. Basic VR Validation

Before debugging the complete surgical simulation, verify the following:

```text
HMD rotation changes Endoscope Camera rotation
HMD translation changes Endoscope Camera position
Unity scene is visible inside the headset
```

If rotation and position remain unchanged, first verify:

- Meta/Oculus runtime status
- HMD and sensor connection
- XR runtime configuration
- GPU selection
- XR simulator disabled

Do not modify `Endoscope Camera` manually while physical tracking is active.

---

## 10. Input Configuration

The Unity project currently uses:

```text
Active Input Handling = Both
```

This is intentional.

The XR system uses the newer Unity Input System, while existing simulation code and the HMD calibration script currently use `UnityEngine.Input`.

Changing the project to only one input backend may break existing functionality.

---

## 11. Relevant Files

```text
Assets/Scenes/DualToolSceneVR.unity
Assets/Scripts/VR/HMDCalibration.cs
Assets/XR/
Packages/manifest.json
Packages/packages-lock.json
ProjectSettings/EditorBuildSettings.asset
ProjectSettings/ProjectSettings.asset
```

The original scene:

```text
Assets/Scenes/DualToolScene.unity
```

is intentionally kept separate from the VR scene.

---

## 12. Recommended Workflow

For VR development:

```text
git checkout vr-integration
git pull origin vr-integration
```

Open:

```text
Assets/Scenes/DualToolSceneVR.unity
```

Then:

```text
1. Start the VR runtime
2. Connect and verify the headset
3. Start Unity
4. Open DualToolSceneVR
5. Press Play
6. Confirm positional and rotational tracking
7. Press C to calibrate the virtual endoscope viewpoint
```

Any future VR-specific scene changes should preferably remain in `DualToolSceneVR` so that the standard `DualToolScene` workflow remains unaffected.
