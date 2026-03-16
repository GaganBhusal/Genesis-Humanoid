## Installation

Copy `changing` folder to G1's onboard Jetson.

```bash
# On local machine
uv pip install minimalmodbus pyserial cyclonedds

# On Jetson
pip install minimalmodbus pyserial cyclonedds
```

## Evaluation

```bash
# On Jetson
ls /dev/ttyUSB*
# If return (empty), install CH341 module on jetson.
# If return /dev/ttyUSB0, the gripper is correctly connected to the robot

sudo usermod -aG dialout $USER
python utils/changingtek_p_rtu_Servo.py
```

## Gripper DDS (server on Jetson, test on your PC)

**On the Jetson (e.g. 192.168.123.164):** test the server (same domain as test, default 0):

```bash
bash scripts/run_gripper_dds_server.bash
```

**On your local machine:** from the repo root, run the test so it discovers the Jetson:

```bash
python gripper_dds_eval.py
```

If the test failed, update the ip address in `cyclonedds.xml` by the result of `ip addr | grep 192.168.123.*` on both local and Jetson.