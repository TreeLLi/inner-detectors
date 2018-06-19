'''
Inner detector matcher:

To match semantic concepts with hidden units in intermediate layers

'''

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.append(root_path)

from utils.helper.data_loader import BatchLoader
from utils.model.model_agent import ModelAgent

from config import CONFIG

if __name__ == "__main__":
    bl = BatchLoader()
    matches = initMatchResults()
    
    while bl:
        batch = bl.nextBatch()
        images = batch.imgs

        model = ModelAgent(model=CONFIG.DIS.MODEL)
        activ_maps = model.getActivMaps(images)
        
        if CONFIG.DIS.REFLECT == "linear":
            from utils.dissection.linear_ref import reflect
        elif CONFIG.DIS.REFLECT == "deconvnet":
            from utils.dissection.deconvnet import reflect

        ref_activ_maps = reflect(activ_maps, model=CONFIG.DIS.MODEL)
        annos = batch.annos

        batch_matches = matchAnnosActivs(annos, ref_activ_maps)
        combineMatches(matches, batch_matches)

    saveMatches(matches)
