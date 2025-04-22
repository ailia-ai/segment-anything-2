export PJRT_DEVICE=CPU
python3 export_image_predictor.py --accuracy int8 --image_size 512 --mode calibration
python3 export_video_predictor.py --accuracy int8 --image_size 512 --mode calibration
python3 export_image_predictor.py --framework tflite --accuracy int8 --image_size 512
python3 export_video_predictor.py --framework tflite --accuracy int8 --image_size 512
