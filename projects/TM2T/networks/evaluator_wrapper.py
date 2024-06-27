from networks.modules import *
from utils.word_vectorizer import POS_enumerator
from os.path import join as pjoin

def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose-4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    if not opt.use_transformers:
        text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
                                      pos_size=opt.dim_pos_ohot,
                                      hidden_size=opt.dim_text_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)

        motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                          hidden_size=opt.dim_motion_hidden,
                                          output_size=opt.dim_coemb_hidden,
                                          device=opt.device)
    else:
        from clip import MotionTransformer, TextTransformer
        text_enc = TextTransformer(embed_dim=512, context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12, device=opt.device)
        motion_enc = MotionTransformer(input_resolution=49, patch_size=1, width=768, layers=12, heads=768//64, output_dim=512, device=opt.device)
        print("use Transformers as Encoder ranther than GRU")
    
    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper(object):

    def __init__(self, opt):

        if opt.dataset_name == 't2m':
            opt.dim_pose = 263
        elif opt.dataset_name == 'kit':
            opt.dim_pose = 251
        else:
            raise KeyError('Dataset not Recognized!!!')

        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512
        try:
            opt.use_transformers = opt.use_transformers
        except:
            print('can not find opt.use_transformers, set opt.use_transformers as True')
            opt.use_transformers = True
            
        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens, caption=None):
        with torch.no_grad():
            if not self.opt.use_transformers:
                word_embs = word_embs.detach().to(self.device).float()
                pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]
#             print(motions.shape)
            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Text Encoding'''
            if not self.opt.use_transformers:
                text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            else:
                text_embedding = self.text_encoder(caption)
#             text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]
            
            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding
