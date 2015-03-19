Config(
    net=Config(
        dec_transition='GatedRecurrent',
        enc_transition='GatedRecurrent',
        attention_type='content_and_cumsum'),
    regularization=Config(
        dropout=True),
    initialization=[
        ("/recognizer", "rec_weights_init", "IsotropicGaussian(0.1)")],
    data=Config(
        normalization="norm.pkl"))
