# Cloned Repo - `InstantID` (Not Working)
```git clone https://github.com/InstantID/InstantID.git```

# Cloned Repo - `PuLID`
```git clone https://github.com/ToTheBeginning/PuLID.git```


For Run IP-Adapter
1.
```mkdir -p /root/arc_3/models/image_encoder```

2.
```cd /root/arc_3/models/image_encoder```

3.
```apt update apt install git-lfs -y git lfs install```

4. 
```git clone https://huggingface.co/h94/IP-Adapter```

5.
```ls /root/arc_3/models/image_encoder/IP-Adapter/models/image_encoder```

6. You Must see
```config.json model.safetensors preprocessor_config.json```

7.
```cd /root/arc_3/models```

8.
```wget https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin```

9.
```/root/arc_3/models/

    ip-adapter-faceid-plusv2_sdxl.bin

    image_encoder/
        IP-Adapter/
            models/
                image_encoder/
                    config.json
                    model.safetensors
                    preprocessor_config.json```

10.
```step_3/pulid_pose_transfer.py```

11.
```ip_model = IPAdapterFaceIDPlus(
    pipe,
    "/root/arc_3/models/image_encoder/IP-Adapter/models/image_encoder",
    "/root/arc_3/models/ip-adapter-faceid-plusv2_sdxl.bin",
    DEVICE
)```

12. Run
```python main.py```