Config(
    net=Config(
        dim_dec=250,
        dim_bidir=250,
        dims_bottom=[250],
        dec_transition='GatedRecurrent',
        enc_transition='GatedRecurrent',
        attention_type='content_and_cumsum',
        use_states_for_readout=True),
    regularization=Config(
        dropout=True),
    initialization=InitList([
        ("/recognizer", "rec_weights_init", "IsotropicGaussian(0.1)")]),
    data=Config(
        sort_k_batches=10,
        normalization="norm.pkl"))
