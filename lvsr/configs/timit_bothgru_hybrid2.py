Config(
    net=Config(dec_transition='GatedRecurrent',
               enc_transition='GatedRecurrent',
               attention_type='hybrid2',
               shift_predictor_dims=[100],
               max_left=10,
               max_right=100),
    initialization=[
        ("/recognizer", "rec_weights_init", "IsotropicGaussian(0.1)"),
        ("/recognizer/generator/att_trans/hybrid_att/loc_att",
         "weights_init", "IsotropicGaussian(0.001)"),
        ("/recognizer/generator/att_trans/hybrid_att/loc_att",
         "biases_init", "IsotropicGaussian(5.0)")],
    data=Config(normalization="norm.pkl"))
