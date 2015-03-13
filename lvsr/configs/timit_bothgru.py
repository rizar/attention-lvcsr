Config(
    net=Config(dec_transition='GatedRecurrent',
               enc_transition='GatedRecurrent'),
    initialization=[
        ("/recognizer", "rec_weights_init", "IsotropicGaussian(0.1)")],
    data=Config(normalization="norm.pkl"))
