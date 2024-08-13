def test_make_motion_template_stress():
    from live_portrait_pipeline import LivePortraitPipeline

    class MockLivePortraitWrapper:
        def get_kp_info(self, img):
            return {'scale': 1, 'pitch': 0, 'yaw': 0, 'roll': 0, 'exp': 0, 't': 0, 'kp': 0}

        def transform_keypoint(self, x_i_info):
            return x_i_info['kp']

    pipeline = LivePortraitPipeline(InferenceConfig(), CropConfig())
    pipeline.live_portrait_wrapper = MockLivePortraitWrapper()

    img_list = [np.zeros((100, 100, 3))] * 1000
    c_eyes_lst = [1] * 1000
    c_lip_lst = [2] * 1000

    template_dict = pipeline.make_motion_template(img_list, c_eyes_lst, c_lip_lst, output_fps=25)

    assert template_dict['n_frames'] == 1000
    assert len(template_dict['motion']) == 1000
    assert len(template_dict['c_eyes_lst']) == 1000
    assert len(template_dict['c_lip_lst']) == 1000
