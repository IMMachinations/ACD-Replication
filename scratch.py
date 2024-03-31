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
class ACDCNode():
    
    def __init__(self, name):
        self.parent_nodes = []
        self.name = name
        self.hidden_input = None
        self.hidden_output = None
        self.frozen = False
        
    def __init__(self, parent_nodes: List, name: str):
        self.parent_nodes = parent_nodes
        self.name = name
        self.hidden_output = None
        self.hidden_input = None
    
    def freeze(self):
        if(self.hidden_input is None):
            raise ValueError("Output not set")
        self.frozen = True
        
    def unfreeze(self):
        self.frozen = False
         
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
        if(self.frozen):
            return self.hidden_input
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
        self.parent_nodes = []
        self.value = value
        self.name = name
    def out(self):
        return self.value
    
    def input(self, shape):
        return self.value
    
    def __str__(self) -> str:
        return f"Constant Node {self.name}, {self.__repr__()}"

def attention_layer_node_input_hook_function(nodelist: List[ACDCNode], hook_value: Tensor, hook: hook_points.HookPoint):
    #print(f"(ATT) Reading from {nodelist} to {hook.name}")
    shape = hook_value[0,:,0,:].shape
    for node in range(len(nodelist)):
        hook_value[0,:,node,:] = nodelist[node].input(shape)
    return hook_value
        
def attention_layer_node_output_hook_function(nodelist: List[ACDCNode], hook_value: Tensor, hook: hook_points.HookPoint):
    #print(f"(ATT) Writing {hook.name} to {nodelist}")
    for node in range(len(nodelist)):
        nodelist[node].set_output(hook_value[0,:,node,:])
        
def mlp_layer_node_input_hook_function(node: ACDCNode, hook_value: Tensor, hook: hook_points.HookPoint):
    #print(f"(MLP) Reading from {hook.name} to {node}")
    shape = hook_value[0,:,:].shape
    hook_value[0,:,:] = node.input(shape)
    return hook_value

def mlp_layer_node_output_hook_function(node: ACDCNode, hook_value: Tensor, hook: hook_points.HookPoint):
    node.set_output(hook_value[0,:,:])
    #print(f"(MLP) Writing {hook.name} to {node}")
    
def embedding_layer_node_output_hook_function(node: ACDCNode, hook_value: Tensor, hook: hook_points.HookPoint):
    node.set_output(hook_value[0,:,:])
    #print(node.hidden_output)
    #print(f"(EMB) Writing {hook.name} to {node}")

def positional_embedding_layer_node_output_hook_function(node: ACDCNode, hook_value: Tensor, hook: hook_points.HookPoint):
    node.set_output(hook_value[0,:,:])
    #print(node.hidden_output)
    #print(f"(POS) Writing {hook.name} to {node}")

def pre_unembedding_layer_node_input_hook_function(node: ACDCNode, hook_value: Tensor, hook: hook_points.HookPoint):
    #print(f"(OUT) Reading from {node} to {hook.name}")
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
    
    output_node = ACDCNode(previous_nodes[:], "Output")
    output_tuple = ("Output", f"blocks.{model.cfg.n_layers-1}.hook_resid_post", output_node)
    previous_nodes.append(output_node)
    
    return embedding_tuple, positional_tuple, node_tuple_list, output_tuple, previous_nodes

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
embedding_tuple, positional_tuple, layer_node_tuple_list, output_tuple, nodes = add_node_scaffold_to_model(model)

hooks = create_hooks(embedding_tuple=embedding_tuple, positional_tuple=positional_tuple, layer_node_tuple_list=layer_node_tuple_list, output_tuple=output_tuple)
#for hook in hooks:
#   print(hook)
hooked_output = model.run_with_hooks("Vernon Dursley and Petunia Durs", fwd_hooks=hooks)
logits, cache = model.run_with_cache("Vernon Dursley and Petunia Durs")

# %%
def prune_node(node: ACDCNode, hooks: List[Tuple[str, Callable]], model: HookedTransformer, prompt: str, prune_value: Float = 1.0):
    for index in range(len(node.parent_nodes) - 1, -1, -1):
        old_logits = model.run_with_hooks("Vernon Dursley and Petunia Durs", fwd_hooks=hooks)
        parent_node = node.parent_nodes.pop(index)
        new_logits = model.run_with_hooks("Vernon Dursley and Petunia Durs", fwd_hooks=hooks)
        KLDivergence_loss = nn.KLDivLoss(reduction='batchmean',log_target=True)(old_logits[0,-1,:], new_logits[0,-1,:])
        print(KLDivergence_loss)