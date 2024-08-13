import ffmpeg

def make_square(input_video_path, output_video_path, crop=False):
    # Load the input video
    input_video = ffmpeg.input(input_video_path)

    # Get the video dimensions
    probe = ffmpeg.probe(input_video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])

    # Determine whether to crop or pad
    if crop:
        # Crop the video to a square aspect ratio
        square_side = min(width, height)
        x_offset = (width - square_side) // 2
        y_offset = (height - square_side) // 2
        video = input_video.video.crop(x_offset, y_offset, square_side, square_side)
    else:
        # Pad the video to a square aspect ratio
        if width > height:
            padding = (width - height) // 2
            video = input_video.video.filter('pad', width, width, 0, padding)
        else:
            padding = (height - width) // 2
            video = input_video.video.filter('pad', height, height, padding, 0)

    # Include the audio stream
    audio = input_video.audio

    # Output the result with audio included
    output = ffmpeg.output(video, audio, output_video_path)
    output.run()

# Example usage
make_square('/home/codemonkeyxl/Downloads/Dubstep Face(1).mp4', 'square_driver3.mp4', crop=False)