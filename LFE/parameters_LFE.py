class Parameters(object):
    def __init__(self):
        self.initialize()
    def initialize(self):
        # The following are parameters or paths that need to be customized according to the actual situation.

        self.model_name =
        self.CUDA_VISIBLE_DEVICES =
        self.i_txt =                             #List of the training samples that recorded in a [.txt] file
        self.img_base =                          #Path of images
        self.GT_lab_base =                       #Path of gound truth
        self.train_lab_base =                    #Path of training labels(partial annotation)
        self.v_txt =                             #List of the verifying samples that recorded in a [.txt] file
        self.v_save_path =                       #Path where the predictions on verifying samples are saved
        self.t_txt =                             #List of the testing samples that recorded in a [.txt] file
        self.t_save_path =                       #Path where the predictions on testing samples are saved
        self.Meanstd_dir =                       #Path of the information of dataset (mean and std) which can be saved as a [.npy] file
        self.checkpoint_dir =                    #Path of checkpoints

        self.patch_shape =
        self.batch_size =
        self.lr =
        self.n_epochs =
        self.Iters =




