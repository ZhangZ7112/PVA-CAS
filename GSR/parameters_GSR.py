class Parameters(object):
    def __init__(self):
        self.initialize()
    def initialize(self):
        self.model_name =
        self.CUDA_VISIBLE_DEVICES =
        self.i_txt =                             #List of the training samples that recorded in a [.txt] file
        self.img_base =                          #Path of images
        self.GT_lab_base =                       #Path of gound truth
        self.partial_annotation_base =           #Path of partial_annotation
        self.pseudo_base =                       #Path of initialized pseudo label
        self.update_pseudo_base =                #Path of updated pseudo label
        self.v_txt =                             #List of the verifying samples that recorded in a [.txt] file
        self.v_save_path =                       #Path where the predictions on verifying samples are saved
        self.t_txt =                             #List of the testing samples that recorded in a [.txt] file
        self.t_save_path =                       #Path where the predictions on testing samples are saved
        self.Meanstd_dir =                       #Path of the information of dataset (mean and std) which can be saved as a [.npy] file
        self.checkpoint_dir =                    #Path of checkpoints
        self.eta_log_path = 'eta_log_{0}.npy'.format(self.model_name) #Path of the log of eta (confidence level)

        self.patch_shape =
        self.batch_size =
        self.lr =
        self.n_epochs =
        self.Iters =







