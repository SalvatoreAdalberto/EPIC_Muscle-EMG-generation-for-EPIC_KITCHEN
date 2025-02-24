from models.TRN import RelationModuleMultiScale, RelationModuleMultiScaleWithClassifier
from models.VideoModel import VideoModel
from models.I3D import I3D
from models.FinalClassifier import Classifier, MLP_late_fusion, action_TRN
from models.actionLSTM import ActionLSTM 
from models.FC_VAE import VariationalAutoencoder as VAE
from models.EMG_classifier import EMG_classifier_parametric as EMG_classifier_parametric
from models.Unimodal_classifier import Unimodal_classifier_parametric as Unimodal_classifier