from django.shortcuts import render
from lib.CameraFeed import camera
from django.http import StreamingHttpResponse

def live(request):
    try:
        cam = camera.VideoCamera()
        return StreamingHttpResponse(camera.gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        pass