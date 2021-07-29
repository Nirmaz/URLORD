base_config = dict(

	content_dim = 100,
	class_dim = 1,

	content_std = 1,
	class_std = 0.3,
	round = True,
	recon_decay = 1,
	seg_decay = 1,
	content_decay = 1e-3,
	class_decay = 1e-4,
	n_adain_layers = 4,
	adain_dim=256,

	perceptual_loss=dict(
		layers=[2, 7, 12, 21, 30]
	),

	train=dict(
		batch_size = 1,
		batch_size_u = 1,
		percent_list = [0.7, 0.3],
		n_epochs=100,
		learning_rate=dict(
			generator=3e-4,
			latent=3e-3,
			min=1e-5
		)
	),

	train_encoders=dict(
		batch_size=128,
		n_epochs=200,

		learning_rate=dict(
			max=1e-4,
			min=1e-5
		)
	),
)

base_config_3d = dict(

	content_dim = 100,
	class_dim = 5,

	content_std = 1,
	class_std = 0.3,
	round = True,
	recon_decay = 5,
	seg_decay = 1,
	content_decay = 1e-3,
	class_decay = 1e-4,
	n_adain_layers = 2,
	adain_dim = 16,

	perceptual_loss=dict(
		layers=[2, 7, 12, 21, 30]
	),

	train=dict(
		batch_size = 1,
		batch_size_u = 1,
		percent_list = [0.7, 0.3],
		n_epochs = 500,
		learning_rate=dict(
			generator=1e-5,
			latent=3e-3,
			min=1e-5
		)
	),

	train_encoders=dict(
		batch_size=128,
		n_epochs=200,

		learning_rate=dict(
			max=1e-4,
			min=1e-5
		)
	),
)

base_config_unet = dict(

	train=dict(
		batch_size = 2,
		percent_list = [0.7, 0.3],
		n_epochs = 500,
		learning_rate=dict(
			generator=1e-5,
			min=1e-5
		)
	),
)
