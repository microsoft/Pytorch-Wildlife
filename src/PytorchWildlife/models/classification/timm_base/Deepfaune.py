"""
This is a Pytorch-Wildlife loader for the DeepFaune classifier.
The original DeepFaune models are available at: https://www.deepfaune.cnrs.fr/en/
Licence: CC BY-SA 4.0
Copyright CNRS 2026
simon.chamaille@cefe.cnrs.fr; vincent.miele@cnrs.fr; gaspard.dussert@cnrs.fr
"""

# Import libraries

from torchvision.transforms.functional import InterpolationMode
from .base_classifier import TIMM_BaseClassifierInference
from ....data import transforms as pw_trans

__all__ = [
    "DeepfauneClassifier",
    "DeepFauneClassifierV15"
]

class DeepfauneClassifier(TIMM_BaseClassifierInference):
    """
    Base detector class for dinov2 classifier. This class provides utility methods
    for loading the model, performing single and batch image classifications, and 
    formatting results. Make sure the appropriate file for the model weights has been 
    downloaded to the "models" folder before running DFNE.
    """
    BACKBONE = "vit_large_patch14_dinov2.lvd142m"
    MODEL_NAME = "deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt"
    IMAGE_SIZE = 182
    CLASS_NAMES={
        'fr': ['bison', 'blaireau', 'bouquetin', 'castor', 'cerf', 'chamois', 'chat', 'chevre', 'chevreuil', 'chien', 'daim', 'ecureuil', 'elan', 'equide', 'genette', 'glouton', 'herisson', 'lagomorphe', 'loup', 'loutre', 'lynx', 'marmotte', 'micromammifere', 'mouflon', 'mouton', 'mustelide', 'oiseau', 'ours', 'ragondin', 'raton laveur', 'renard', 'renne', 'sanglier', 'vache'],
        'en': ['bison', 'badger', 'ibex', 'beaver', 'red deer', 'chamois', 'cat', 'goat', 'roe deer', 'dog', 'fallow deer', 'squirrel', 'moose', 'equid', 'genet', 'wolverine', 'hedgehog', 'lagomorph', 'wolf', 'otter', 'lynx', 'marmot', 'micromammal', 'mouflon', 'sheep', 'mustelid', 'bird', 'bear', 'nutria', 'raccoon', 'fox', 'reindeer', 'wild boar', 'cow'],
        'it': ['bisonte', 'tasso', 'stambecco', 'castoro', 'cervo', 'camoscio', 'gatto', 'capra', 'capriolo', 'cane', 'daino', 'scoiattolo', 'alce', 'equide', 'genetta', 'ghiottone', 'riccio', 'lagomorfo', 'lupo', 'lontra', 'lince', 'marmotta', 'micromammifero', 'muflone', 'pecora', 'mustelide', 'uccello', 'orso', 'nutria', 'procione', 'volpe', 'renna', 'cinghiale', 'mucca'],
        'de': ['Bison', 'Dachs', 'Steinbock', 'Biber', 'Rothirsch', 'Gämse', 'Katze', 'Ziege', 'Rehwild', 'Hund', 'Damwild', 'Eichhörnchen', 'Elch', 'Equide', 'Ginsterkatze', 'Vielfraß', 'Igel', 'Lagomorpha', 'Wolf', 'Otter', 'Luchs', 'Murmeltier', 'Kleinsäuger', 'Mufflon', 'Schaf', 'Marder', 'Vogel', 'Bär', 'Nutria', 'Waschbär', 'Fuchs', 'Rentier', 'Wildschwein', 'Kuh'],
    }


    def __init__(self, weights=None, device="cpu", transform=None, class_name_lang='en'):
        url = 'https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.3/deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt'
        self.CLASS_NAMES = {i: c for i, c in enumerate(self.CLASS_NAMES[class_name_lang])}
        if transform is None:
            transform = pw_trans.Classification_Inference_Transform(target_size=self.IMAGE_SIZE, 
                                                                    interpolation=InterpolationMode.BICUBIC, 
                                                                    max_size=None,
                                                                    antialias=None)
        super(DeepfauneClassifier, self).__init__(weights=weights, device=device, url=url, transform=transform,
                                                  weights_key='state_dict', weights_prefix='base_model.')

class DeepFauneClassifierV15(TIMM_BaseClassifierInference):
    """
    Classifier model of DeepFaune v1.5
    """
    BACKBONE = "vit_large_patch16_dinov3.lvd1689m"
    MODEL_NAME = "deepfaune-vit_large_patch16_dinov3.lvd1689m.pt"
    IMAGE_SIZE = 224
    CLASS_NAMES={
            'fr': ['bison', 'blaireau', 'bouquetin', 'castor', 'cerf', 'chacal doré', 'chamois', 'chat', 'chevre',
                'chevreuil', 'chien', 'chien viverrin', 'daim', 'ecureuil', 'elan', 'equide', 'genette', 'glouton',
                'herisson', 'lagomorphe', 'loup', 'loutre', 'lynx', 'mangouste', 'marmotte', 'micromammifere', 'mouflon', 'mouton',
                'mustelide', 'oiseau', 'ours', 'porcepic', 'ragondin', 'rat musqué', 'raton laveur', 'renard',
                'renard arctique', 'renne', 'sanglier', 'vache'],
            'en': ['bison', 'badger', 'ibex', 'beaver', 'red deer', 'golden jackal', 'chamois', 'cat', 'goat',
                'roe deer', 'dog', 'raccoon dog', 'fallow deer', 'squirrel', 'moose', 'equid', 'genet',
                'wolverine', 'hedgehog', 'lagomorph', 'wolf', 'otter', 'lynx', 'mongoose', 'marmot', 'micromammal', 
                'mouflon', 'sheep', 'mustelid', 'bird', 'bear', 'porcupine', 'nutria', 'muskrat', 'raccoon',
                'fox', 'arctic fox', 'reindeer', 'wild boar', 'cow'],
            'it': ['bisonte', 'tasso', 'stambecco', 'castoro', 'cervo', 'sciacallo dorato', 'camoscio', 'gatto', 'capra',
                'capriolo', 'cane', 'cane procione', 'daino', 'scoiattolo', 'alce', 'equide', 'genetta', 'ghiottone',
                'riccio', 'lagomorfo', 'lupo', 'lontra', 'lince', 'mangusta', 'marmotta', 'micromammifero', 'muflone', 'pecora',
                'mustelide', 'uccello', 'orso', 'istrice', 'nutria', 'ondatra', 'procione', 'volpe',
                'volpe artica', 'renna', 'cinghiale', 'mucca'],      
            'de': ['Bison', 'Dachs', 'Steinbock', 'Biber', 'Rothirsch', 'Goldschakal', 'Gämse', 'Katze', 'Ziege',
                'Rehwild', 'Hund', 'Marderhund', 'Damwild', 'Eichhörnchen', 'Elch', 'Equide', 'Ginsterkatze',
                'Vielfraß', 'Igel', 'Lagomorpha', 'Wolf', 'Otter', 'Luchs', 'Manguste', 'Murmeltier', 'Kleinsäuger', 'Mufflon',
                'Schaf', 'Marder', 'Vogel', 'Bär', 'Stachelschwein', 'Nutria', 'Bisamratte', 'Waschbär', 'Fuchs',
                'Polarfuchs', 'Rentier', 'Wildschwein', 'Kuh'],
            'es': ['bisonte', 'tejón', 'cabra montés', 'castor', 'ciervo rojo', 'chacal dorado', 'rebeco', 'gato', 'cabra',
                'corzo', 'perro', 'perro mapache', 'gamo', 'ardilla', 'alce', 'équido', 'gineta', 'glotón', 'erizo', 'lagomorfo',
                'lobo', 'nutria', 'lince', 'mangosta', 'marmota', 'micromamífero', 'muflón', 'oveja', 'mustélido', 'ave', 'oso', 'puercoespín',
                'coipú', 'rata almizclera', 'mapache', 'zorro', 'zorro ártico', 'reno', 'jabalí', 'vaca'],    
            'no': ['bison', 'grevling', 'steinbukk', 'bever', 'hjort', 'gullsjakal', 'gemse', 'katt', 'geit',
                'rådyr', 'hund', 'mårhund', 'dåhjort', 'ekorn', 'elg', 'hestedyr', 'genett', 'jerv',
                'pinnsvin', 'hare', 'ulv', 'oter', 'gaupe', 'mangust', 'murmeldyr', 'småpattedyr', 'muflon', 'sau',
                'mårdyr', 'fugl', 'bjørn', 'piggsvin', 'beverrotte', 'bisamrotte', 'vaskebjørn', 'rev',
                'fjellrev', 'rein', 'villsvin', 'ku'],  
            'se': ['bison', 'grävling', 'stenbock', 'bäver', 'kronhjort', 'guldschakal', 'gems', 'katt', 'get',
                'rådyr', 'hund', 'mårdhund', 'dovhjort', 'ekorre', 'älg', 'hästdjur', 'genett', 'järv',
                'igelkott', 'hare', 'varg', 'utter', 'lo', 'mangust', 'murmeldjur', 'smådäggdjur', 'mufflonfår', 'får',
                'mårdjur', 'fågel', 'björn', 'piggsvin', 'sumpbäver', 'bisam', 'tvättbjörn', 'räv',
                'fjällräv', 'ren', 'vildsvin', 'ko'],
            'pt': ['bisão', 'texugo', 'íbex', 'castor', 'veado', 'chacal-dourado', 'camurça', 'gato', 'cabra',
                'corço', 'cão', 'cão-guaxinim', 'gamo', 'esquilo', 'alce', 'equídeo', 'gineta', 'glutão', 
                'ouriço', 'lagomorfo', 'lobo', 'lontra', 'lince', 'mangusto', 'marmota', 'micromamífero', 'muflão', 
                'ovelha', 'mustelídeo', 'ave', 'urso', 'porco-espinho', 'nutria', 'rato-almiscarado', 
                'guaxinim', 'raposa', 'raposa-do-ártico', 'rena', 'javali', 'vaca']
        }


    def __init__(self, weights=None, device="cpu", transform=None, class_name_lang='en'):
        url = 'https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.5/deepfaune-vit_large_patch16_dinov3.lvd1689m.pt'
        self.CLASS_NAMES = {i: c for i, c in enumerate(self.CLASS_NAMES[class_name_lang])}
        if transform is None:
            transform = pw_trans.Classification_Inference_Transform(target_size=self.IMAGE_SIZE, 
                                                                    interpolation=InterpolationMode.BICUBIC, 
                                                                    max_size=None,
                                                                    antialias=None)
        super(DeepFauneClassifierV15, self).__init__(weights=weights, device=device, url=url, transform=transform,
                                                  weights_key='state_dict', weights_prefix='base_model.', )

