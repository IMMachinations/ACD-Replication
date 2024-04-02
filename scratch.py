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
    
    def __init__(self, name: str):
        self.parent_nodes = []
        self.corrupted_parents = []
        self.name = name
        self.hidden_input = None
        self.hidden_output = None
        self.frozen = False
        self.corrupted_output = {}
        self.current_key = None
        
    def __init__(self, parent_nodes: List, name: str):
        self.parent_nodes = parent_nodes
        self.corrupted_parents = []
        self.name = name
        self.hidden_output = None
        self.hidden_input = None
        self.frozen = False
        self.corrupted_output = {}
        self.current_key = None
    
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
    
    def corrupted_out(self, key: str):
        if(self.corrupted_output[key] is None):
            raise ValueError("Output not set")
        return self.corrupted_output[key]
    
    def set_corrupted_output(self, key: str, output: Tensor):
        self.corrupted_output[key] = output
    
    def save_corrupt_output(self, key: str):
        if(self.hidden_output is None):
            raise ValueError("Output not set")
        self.corrupted_output[key] = self.hidden_output
        return self.hidden_output
    
    def input(self, shape, key: str = None):
        if(self.frozen):
            return self.hidden_input
        node_input = t.zeros(shape)
        for parent in self.parent_nodes:
            node_input += parent.out()
        if key is not None:
            for parent in self.corrupted_parents:
                node_input += parent.corrupted_out(key)
        else:
            for parent in self.corrupted_parents:
                try:
                    node_input += parent.corrupted_out(self.current_key)
                except BaseException as e:
                    print(self.name)
                    print(self.current_key)
                    print(parent.name)
                    print(parent.corrupted_output.keys())
                    for key, tensor in parent.corrupted_output.items():
                        print(f"{key}: {tensor.shape}")
                    raise e
                    
        self.hidden_input = node_input
        return node_input
    
    def __str__(self) -> str:
        return f"Node {self.name}, {self.__repr__()}"
    
    
class ConstantNode(ACDCNode):
    def __init__(self, value: Tensor, name: str):
        self.parent_nodes = []
        self.value = value
        self.name = name
        self.corrupted_values = {}
        self.current_key = None

    def out(self):
        return self.value
    
    def set_corrupted_output(self, key: str, output: Tensor):
        self.corrupted_values[key] = output
        
    def corrupted_out(self, key: str):
        if key in self.corrupted_values.keys():
            return self.corrupted_values[key]
        return self.value
    
    def save_corrupt_output(self, key: str):
        if(False):
            self.corrupted_values[key] = self.value
    
    def input(self, shape, key: str = None):
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
def zero_prune_node(node: ACDCNode, hooks: List[Tuple[str, Callable]], model: HookedTransformer, prompts: List[str], prune_value: Float = 1.0):
    base_logits = t.zeros((len(prompts), model.cfg.d_vocab))
    base_logits = nn.Softmax(dim=-1)(base_logits)
    for index, prompt in enumerate(prompts):
        base_logits[index,:] =  model.run_with_hooks(prompt, fwd_hooks=hooks)[0,-1,:]
    
    for index in range(len(node.parent_nodes) - 1, -1, -1):
        old_logits = t.zeros((len(prompts), model.cfg.d_vocab))
        for index, prompt in enumerate(prompts):
            old_logits[index,:] = model.run_with_hooks(prompt, fwd_hooks=hooks)[0,-1,:]
        old_logits = nn.Softmax(dim=-1)(old_logits)
        parent_node = node.parent_nodes.pop(index)
        new_logits = t.zeros((len(prompts), model.cfg.d_vocab))
        for index, prompt in enumerate(prompts):
            new_logits[index,:] = model.run_with_hooks(prompt, fwd_hooks=hooks)[0,-1,:]
        new_logits = nn.Softmax(dim=-1)(new_logits)
        old_KLDivergence_loss = nn.KLDivLoss(reduction='batchmean',log_target=True)(old_logits, base_logits)
        new_KLDivergence_loss = nn.KLDivLoss(reduction='batchmean',log_target=True)(new_logits, base_logits)
        
        print(new_KLDivergence_loss - old_KLDivergence_loss)
        if(new_KLDivergence_loss - old_KLDivergence_loss >= prune_value):
            node.parent_nodes.append(parent_node)
            
def alternate_run_prune_node(node: ACDCNode, all_nodes: List[ACDCNode], hooks: List[Tuple[str, Callable]], model: HookedTransformer, prompts: List[Tuple[str,str]], prune_value: Float = 1.0):
    kl_divergence_losses = []
    base_logits = t.zeros((len(prompts), model.cfg.d_vocab))
    base_logits = nn.Softmax(dim=-1)(base_logits)
    for index, prompt in enumerate(prompts):
        for iter_node in all_nodes:
            iter_node.current_key = prompt[0]
        base_logits[index,:] =  model.run_with_hooks(prompt[0], fwd_hooks=hooks)[0,-1,:]
        for iter_node in all_nodes[:-1]:
            iter_node.save_corrupt_output(prompt[0])

    for index in tqdm(range(len(node.parent_nodes) - 1, -1, -1)):
        
        old_logits = t.zeros((len(prompts), model.cfg.d_vocab))
        for i, prompt in enumerate(prompts):
            for iter_node in all_nodes:
                iter_node.current_key = prompt[0]
            node.current_key = prompt[0]
            old_logits[i,:] = model.run_with_hooks(prompt[1], fwd_hooks=hooks)[0,-1,:]
        old_logits = nn.Softmax(dim=-1)(old_logits)
        node.corrupted_parents.append(node.parent_nodes.pop(index))
        
        new_logits = t.zeros((len(prompts), model.cfg.d_vocab))
        for i, prompt in enumerate(prompts):
            for iter_node in all_nodes:
                iter_node.current_key = prompt[0]
            node.current_key = prompt[0]
            new_logits[i,:] = model.run_with_hooks(prompt[1], fwd_hooks=hooks)[0,-1,:]
        new_logits = nn.Softmax(dim=-1)(new_logits)
        
        old_KLDivergence_loss = nn.KLDivLoss(reduction='batchmean',log_target=True)(old_logits, base_logits)
        new_KLDivergence_loss = nn.KLDivLoss(reduction='batchmean',log_target=True)(new_logits, base_logits)
        
        kl_difference = t.abs(new_KLDivergence_loss - old_KLDivergence_loss)
        kl_divergence_losses.append(kl_difference.detach())
        if(kl_difference >= prune_value):
            node.parent_nodes.append(node.corrupted_parents.pop())
            #print(list(map(str, node.corrupted_parents)))
    return kl_divergence_losses
            
# %%
prompts = [("Vernon Thornefield and Petunia Durs", "Vernon Dursley and Petunia Durs"), 
           ("The second largest is Saint Paul. In Minnesota, Dul", "The second largest is Duluth. In Minnesota, Dul"),
           ("The capital of Minnesota is in Saint Paul. The patron saint of missionaries is St", "The capital of Minnesota is St. Paul. The patron saint of missionaries is St" ),
           ("Jan and Clementine went to the beach. Cos","Jan and Cosette went to the beach. Cos"),
           ("My name is John Smith. I am Cal", "My name is Calista. I am Cal") ]

model = HookedTransformer.from_pretrained("gpt2-small")
model.cfg.use_attn_in = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
embedding_tuple, positional_tuple, layer_node_tuple_list, output_tuple, nodes = add_node_scaffold_to_model(model)

hooks = create_hooks(embedding_tuple=embedding_tuple, positional_tuple=positional_tuple, layer_node_tuple_list=layer_node_tuple_list, output_tuple=output_tuple)

for i in range(len(nodes) - 1, -1, -1):
    is_relevant = False
    for node in nodes:
        if node[i] in node.parent_nodes:
            is_relevant = True
            break
    
    if is_relevant:        
        alternate_run_prune_node(nodes[i], nodes, hooks, model, prompts, prune_value=5000000)

#first_losses = alternate_run_prune_node(nodes[-1], nodes, hooks, model, prompts, prune_value=500000)
#%%
#second_losses = alternate_run_prune_node(nodes[-2], nodes, hooks, model, prompts, prune_value=500000)
# %%
#third_losses = alternate_run_prune_node(nodes[-3], nodes, hooks, model, prompts, prune_value=500000)
# %%
for prompt in prompts:
    
    print(len(model.to_str_tokens(prompt[0])) == len(model.to_str_tokens(prompt[1])) )
    print(len(model.to_str_tokens(prompt[0])))
    print(len(model.to_str_tokens(prompt[1])))
    print(model.to_str_tokens(prompt[0]))
    print(model.to_str_tokens(prompt[1]))
# %%
