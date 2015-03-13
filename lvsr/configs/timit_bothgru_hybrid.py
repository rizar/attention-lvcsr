Config(
    net=Config(dec_transition='GatedRecurrent',
               enc_transition='GatedRecurrent',
               attention_type='hybrid',
               shift_predictor_dims=[100],
               max_left=10,
               max_right=100),
    initialization=[
        ("/recognizer", "rec_weights_init", "IsotropicGaussian(0.1)")],
    data=Config(normalization="norm.pkl"))
