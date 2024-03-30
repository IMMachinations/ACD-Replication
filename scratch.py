#%%
import torch as t
from torch import nn, Tensor
from torch.nn import functional as F
from dataclasses import dataclass
import time
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import einops
import plotly.express as px
from pathlib import Path
from jaxtyping import Float
from typing import Optional, Callable, Union, List, Tuple
from tqdm.auto import tqdm
from dataclasses import dataclass
from transformer_lens import HookedTransformer, hook_points
from functools import partial



device = t.device("cuda" if t.cuda.is_available() else "cpu")
#%%
model = HookedTransformer.from_pretrained("gpt2-small")
#%%
print(model.cfg.n_layers)
print(model.cfg.n_heads)
print(f"W_O: {model.W_O.shape}")
print(f"W_V: {model.W_V.shape}")
print(f"W_Q: {model.W_Q.shape}")
print(f"W_K: {model.W_K.shape}")
print(f"b_O: {model.b_O.shape}")
print(f"b_V: {model.b_V.shape}")
print(f"b_Q: {model.b_Q.shape}")
print(f"b_K: {model.b_K.shape}")
#%%
logits = model("This is the end of the sentence")
print(model.to_string(t.argmax(logits[0,-1,:])))
logits = model("Is this the end of the sentence")
print(model.to_string(t.argmax(logits[0,-1,:])))
#%%
for _ in range(10):
    logits = model("1+2=")
    print(model.to_string(t.argmax(logits[0,-1,:])))
# %%
logits = model("You are going to the store")
print(model.to_string(t.argmax(logits[0,-1,:])))
logits = model("Are you going to the store")
print(model.to_string(t.argmax(logits[0,-1,:])))

#%%
class ACDCNode():
    
    def __init(self, name):
        self.parent_nodes = []
        self.name = name
        self.hidden_input = None
        self.hidden_output = None
        
    def __init__(self, parent_nodes: List, name: str):
        self.parent_nodes = parent_nodes
        self.name = name
        self.hidden_output = None
        self.hidden_input = None
    
    def set_output(self, output: Tensor):
        self.hidden_output = output
    
    def clear_output(self):
        self.hidden_output = None
        
    def out(self):
        if(self.hidden_output is None):
            print(self)
            raise ValueError("Output not set")
        return self.hidden_output

    def input(self, shape):
        #if(self.hidden_input):
        #    if(self.hidden_input.shape != shape):
        #        raise ValueError("Shape mismatch")
        #    return self.hidden_input
        input = t.zeros(shape)
        for parent in self.parent_nodes:
            input += parent.out()
        self.hidden_input = input
        return input
    
    def __str__(self) -> str:
        return f"Node {self.name}, {self.__repr__()}"
    #def __repr__(self) -> str:

    #    return self.name
    
class ConstantNode(ACDCNode):
    def __init__(self, value: Tensor, name: str):
        self.value = value
        self.name = name
    def out(self):
        return self.value
    
    def input(self, shape):
        return self.value
    
    def __str__(self) -> str:
        return f"Constant Node {self.name}, {self.__repr__()}"

def attention_layer_node_input_hook_function(nodelist: List[ACDCNode], hook_value: Tensor, hook: hook_points.HookPoint):
    print(f"(ATT) Reading from {nodelist} to {hook.name}")
    shape = hook_value[0,:,0,:].shape
    for node in range(len(nodelist)):
        hook_value[0,:,node,:] = nodelist[node].input(shape)
    return hook_value
        
def attention_layer_node_output_hook_function(nodelist: List[ACDCNode], hook_value: Tensor, hook: hook_points.HookPoint):
    print(f"(ATT) Writing {hook.name} to {nodelist}")
    for node in range(len(nodelist)):
        nodelist[node].set_output(hook_value[0,:,node,:])
        
def mlp_layer_node_input_hook_function(node: ACDCNode, hook_value: Tensor, hook: hook_points.HookPoint):
    print(f"(MLP) Reading from {hook.name} to {node}")
    shape = hook_value[0,:,:].shape
    hook_value[0,:,:] = node.input(shape)
    return hook_value

def mlp_layer_node_output_hook_function(node: ACDCNode, hook_value: Tensor, hook: hook_points.HookPoint):
    node.set_output(hook_value[0,:,:])
    print(f"(MLP) Writing {hook.name} to {node}")
    
def embedding_layer_node_output_hook_function(node: ACDCNode, hook_value: Tensor, hook: hook_points.HookPoint):
    node.set_output(hook_value[0,:,:])
    #print(node.hidden_output)
    print(f"(EMB) Writing {hook.name} to {node}")

def positional_embedding_layer_node_output_hook_function(node: ACDCNode, hook_value: Tensor, hook: hook_points.HookPoint):
    node.set_output(hook_value[0,:,:])
    #print(node.hidden_output)
    print(f"(POS) Writing {hook.name} to {node}")

def pre_unembedding_layer_node_input_hook_function(node: ACDCNode, hook_value: Tensor, hook: hook_points.HookPoint):
    print(f"(OUT) Reading from {node} to {hook.name}")
    shape = hook_value[0,:,:].shape
    hook_value[0,:,:] = node.input(shape)
    return hook_value

# %%
def add_node_scaffold_to_model(model: HookedTransformer):
    node_tuple_list = []
    previous_nodes = []
    
    embedding_tuple = ("Embedding", "hook_embed", ACDCNode([], "Embedding"))
    
    
    positional_tuple = ("Positional Embedding", "hook_pos_embed", ACDCNode([], "Positional Embedding"))
    
    previous_nodes.append(embedding_tuple[-1])
    #print(previous_nodes)
    previous_nodes.append(positional_tuple[-1])
    #print(previous_nodes)
    for layer in range(model.cfg.n_layers):
        attention_layer = []
        for head in range(model.cfg.n_heads):
            attention_layer.append(ACDCNode(previous_nodes[:], f"Layer {layer} Head {head}"))
        
        node_tuple_list.append(("Attention", f"blocks.{layer}.hook_attn_in", f"blocks.{layer}.attn.hook_result", list(attention_layer)))
                
        for head in attention_layer:
            previous_nodes.append(head)
            
        previous_nodes.append(ConstantNode(model.b_O[layer,:], f"Layer {layer} b_O"))
        
        mlp_layer = ACDCNode(previous_nodes[:], f"Layer {layer} MLP")
        
        node_tuple_list.append(("MLP", f"blocks.{layer}.hook_mlp_in", f"blocks.{layer}.hook_mlp_out", mlp_layer))
        
        previous_nodes.append(mlp_layer)
    
    output_node = ACDCNode(previous_nodes, "Output")
    output_tuple = ("Output", f"blocks.{model.cfg.n_layers-1}.hook_resid_post", output_node)
    
    return embedding_tuple, positional_tuple, node_tuple_list, output_tuple

def create_hooks(embedding_tuple: Tuple[str,str,str,ACDCNode], positional_tuple: Tuple[str,str,str,ACDCNode], layer_node_tuple_list: List[Tuple[str, str, str, List[ACDCNode]]], output_tuple: Tuple[str,str,str,ACDCNode]):   
    hooks = []
    
    hooks.append((embedding_tuple[1], partial(embedding_layer_node_output_hook_function, embedding_tuple[-1])))
        
    hooks.append((positional_tuple[1], partial( positional_embedding_layer_node_output_hook_function, positional_tuple[-1])))
        
    for layer in layer_node_tuple_list:
        if(len(layer) != 4):
            raise ValueError("Node tuple must have 4 elements")
        
        if(layer[0] == "Attention"):
            attn_in_hook_func = partial(attention_layer_node_input_hook_function, layer[-1])
            attn_out_hook_func = partial(attention_layer_node_output_hook_function, layer[-1])
            hooks.append((layer[1], attn_in_hook_func))
            hooks.append((layer[2], attn_out_hook_func))
        
        elif(layer[0] == "MLP"):
            mlp_in_hook_func = partial(mlp_layer_node_input_hook_function, layer[-1])
            mlp_out_hook_func = partial(mlp_layer_node_output_hook_function, layer[-1])
            hooks.append((layer[1], mlp_in_hook_func))
            hooks.append((layer[2], mlp_out_hook_func))
        else:
            raise ValueError("Layer type not recognized")
    
    hooks.append((output_tuple[1], partial(pre_unembedding_layer_node_input_hook_function, output_tuple[-1])))
    
    return hooks
        
# %%
model = HookedTransformer.from_pretrained("gpt2-small")
model.cfg.use_attn_in = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
embedding_tuple, positional_tuple, layer_node_tuple_list, output_tuple = add_node_scaffold_to_model(model)

hooks = create_hooks(embedding_tuple=embedding_tuple, positional_tuple=positional_tuple, layer_node_tuple_list=layer_node_tuple_list, output_tuple=output_tuple)
for hook in hooks:
    print(hook)
hooked_output = model.run_with_hooks("The couch is from Vernon Dursley and Petunia Durs", fwd_hooks=hooks)
logits, cache = model.run_with_cache("The couch is from Vernon Dursley and Petunia Durs")


# %%
class ACDCConfig:
    def __init__(self):
        pass
    
def copy_hooked_transformer(model: HookedTransformer):
    new_model = HookedTransformer(model.cfg, model.tokenizer)
    new_model.load_state_dict(model.state_dict())
    return new_model
    
class HackyACDCHookedTransformer():
    def __init__(self, model: HookedTransformer, config: ACDCConfig):
        self.model = copy_hooked_transformer(model)
        self.config = config
    def zero_ablate_head_hook_function(self,head: int, hook_value: Tensor, hook: hook_points.HookPoint):
        hook_value[0,:,head,:] = 0
        return hook_value
        
    def prune_head(self, layer: int, head: int):
        self.model.W_O[layer,head,:,:] = 0
        self.model.W_V[layer,head,:,:] = 0
        self.model.W_Q[layer,head,:,:] = 0
        self.model.W_K[layer,head,:,:] = 0
        #self.model.b_O[layer,head,:] = 0
        self.model.b_V[layer,head,:] = 0
        self.model.b_Q[layer,head,:] = 0
        self.model.b_K[layer,head,:] = 0
        return self.model
    def prune_heads_zeros(self, inputs: List[str], answers: List[str], threshold: float = 0.5):
        if(len(inputs) != len(answers)):
            raise ValueError("Input and answer must be the same length")
        if(len(inputs) == 0):
            raise ValueError("Input and answer must not be empty")
        logit_difs = np.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads))
        heads = []
        
        attn_result = self.model.cfg.use_attn_result
        self.model.cfg.use_attn_result = True
        
        answer_tokens = [self.model.to_tokens(answer, prepend_bos=False).tolist()[0][0] for answer in answers]
        
        for layer in tqdm(range(self.model.cfg.n_layers - 1, -1, -1)):
            original_answer_logits = np.zeros(len(answer_tokens))
            for input in range(len(inputs)):
                normal_logits = self.model(inputs[input])
                original_answer_logits[input] = normal_logits[0,-1,answer_tokens[input]].detach().cpu().numpy()
            for head in tqdm(range(self.model.cfg.n_heads)):
                new_logits = np.zeros(len(answer_tokens))
                for input in range(len(inputs)):
                    new_logits[input] = self.model.run_with_hooks(inputs[input], fwd_hooks=[(f"blocks.{layer}.attn.hook_result", lambda hook_value, hook : self.zero_ablate_head_hook_function(head, hook_value, hook))])[0,-1,answer_tokens[input]]
                logit_dif = (new_logits - original_answer_logits).mean()
                #print(logit_dif)
                logit_difs[layer, head] = logit_dif
                if np.abs(logit_dif) < threshold:
                    self.prune_head(layer, head)
                else:
                    heads.append((layer, head))
        
        self.model.cfg.use_attn_result = attn_result
        return heads, logit_difs
    def prune_heads_zeroablation_kldivergence(self, inputs: List[str], answers: List[str], threshold: float = 0.5):
        if(len(inputs) != len(answers)):
            raise ValueError("Input and answer must be the same length")
        if(len(inputs) == 0):
            raise ValueError("Input and answer must not be empty")
        num_inputs = len(inputs)
        KLdivergence_differences = np.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads))
        heads = []
        
        attn_result = self.model.cfg.use_attn_result
        self.model.cfg.use_attn_result = True
        
        answer_tokens = [self.model.to_tokens(answer, prepend_bos=False).tolist()[0][0] for answer in answers]
        vocab_size = self.model.cfg.d_vocab_out
        if(vocab_size == -1):
            vocab_size = self.model.cfg.d_vocab
        default_probs = t.zeros(num_inputs,vocab_size)
        for input in range(num_inputs):
            default_probs[input,:] = nn.Softmax(dim=0)(self.model(inputs[input])[0,-1,:])
        current_probs = default_probs
        for layer in tqdm(range(self.model.cfg.n_layers - 1, -1, -1)):
            
            #for input in range(len(inputs)):
            #    normal_logits = self.model(inputs[input])
            #    original_answer_logits[input,:] = nn.Softmax(dim=0)(normal_logits[0,-1,:])
                
            for head in tqdm(range(self.model.cfg.n_heads)):
                new_logprobs = t.zeros((num_inputs,vocab_size))
                for input in range(len(inputs)):
                    new_logprobs[input,:] = nn.Softmax(dim=0)(self.model.run_with_hooks(inputs[input], fwd_hooks=[(f"blocks.{layer}.attn.hook_result", lambda hook_value, hook : self.zero_ablate_head_hook_function(head, hook_value, hook))])[0,-1,:])
                KLdivergence_difference = nn.KLDivLoss(reduction='batchmean',log_target=True)(default_probs, new_logprobs) - nn.KLDivLoss(reduction='batchmean',log_target=True)(default_probs, current_probs)
                #print(KLdivergence_difference)
                KLdivergence_differences[layer,head] = KLdivergence_difference
                if KLdivergence_difference < threshold:
                    self.prune_head(layer, head)
                    current_probs = new_logprobs
                else:
                    heads.append((layer, head))
        
        self.model.cfg.use_attn_result = attn_result
        return heads, KLdivergence_differences
    
    def attention_heads_direct_input_selection_hooks(self, cache, edge_list, hook_value, hook):
        for head in range(len(edge_list)):
            for edge in edge_list[head]:
                hook_value[0,:,head,:] -= cache[f'blocks.{edge[0]}.attn.hook_result'][0,:,edge[1],:]
            return hook_value
    
    def prune_head_edges_zero_ablation_KLdivergence(self, inputs: List[str], answers: List[str], threshold: float = 0.5):
        if(len(inputs) != len(answers)):
            raise ValueError("Input and answer must be the same length")
        if(len(inputs) == 0):
            raise ValueError("Input and answer must not be empty")
        num_inputs = len(inputs)
        answer_tokens = [self.model.to_tokens(answer, prepend_bos=False).tolist()[0][0] for answer in answers]
        vocab_size = self.model.cfg.d_vocab_out
        if(vocab_size == -1):
            vocab_size = self.model.cfg.d_vocab
        
        attn_result = self.model.cfg.use_attn_result
        self.model.cfg.use_attn_result = True
        attn_in = self.model.cfg.use_attn_in
        self.model.cfg.use_attn_in = True
        
        default_probs = t.zeros(num_inputs,vocab_size)
        for input in range(num_inputs):
            default_probs[input,:] = nn.Softmax(dim=0)(self.model(inputs[input])[0,-1,:])
        current_probs = default_probs
        
        base_caches = []
        for input in range(num_inputs):
            base_caches.append(self.model.run_with_cache(inputs[input])[1])
        
        edges = [([] for _ in range(self.model.cfg.n_heads)) for _ in range(self.model.cfg.n_layers)]
        
        for layer in range(self.model.cfg.n_layers):
        for layer in tqdm(range(self.model.cfg.n_layers - 1, -1, -1)):
            for head in tqdm(range(self.model.cfg.n_heads)):
                new_caches = []
                for input in range(num_inputs):
                    new_caches.append(self.model.run_with_cache(inputs[input], fwd_hooks=[(f"blocks.{layer}.attn.hook_result", lambda hook_value, hook : self.zero_ablate_head_hook_function(head, hook_value, hook))])[1])
                KLdivergence_difference = 0
                for input in range(num_inputs):
                    KLdivergence_difference += nn.KLDivLoss(reduction='batchmean',log_target=True)(base_caches[input], new_caches[input])
                KLdivergence_difference /= num_inputs
                if KLdivergence_difference < threshold:
                    self.prune_head(layer, head)
                else:
                    heads.append((layer, head))
        


# %%
cfg = ACDCConfig()
hacked_model = HackyACDCHookedTransformer(model, cfg)


# %%
induction_prompts = ["Vernon Dursley and Petunia Durs","Harry Potter and James", "Disestablishmentarianism is spelled Dis",
                     "Patton Oswalt's full name is Patton Peter Osw", "Patton Oswalt's stage name is alse Patton"]
induction_answers = ["ley", "Potter", "establishment", "alt", "Osw"]
# %%
heads, logit_difs = hacked_model.prune_heads_zeros(induction_prompts, induction_answers, threshold=0.5)
# %%
px.imshow(logit_difs,  labels=dict(x="Head", y="Layer", color="Logit Dif"))
# %%
heads, logit_difs = hacked_model.prune_heads_zeroablation_kldivergence(induction_prompts, induction_answers, threshold=0.05) 
# %%
