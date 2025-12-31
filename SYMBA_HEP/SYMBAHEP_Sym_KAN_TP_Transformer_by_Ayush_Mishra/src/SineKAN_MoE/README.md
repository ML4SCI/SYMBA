# SineKAN-MoE: Mixture of Experts with Sinusoidal KAN's 

**SineKAN-MoE** is the core architectural innovation of this repository.  
It combines two powerful ideas to replace the traditional **Feed Forward Network (FFN)** block in Transformers:

- **SineKAN (Kolmogorov–Arnold Networks with sinusoidal activations)**  
- **Mixture-of-Experts (MoE)**  

This design significantly improves the Transformer’s ability to handle **symbolic and physics-related tasks**, such as predicting **squared amplitudes** in High Energy Physics.

---

###  How SineKAN-MoE Works
1. Input tokens → Transformer embedding + Multi-Head Attention.  
2. Instead of a single FFN, tokens go to **N SineKAN experts**.  
3. A **routing matrix (gating network)** decides which expert(s) each token should use.  
4. Each expert is a **SineKAN network**, trained to specialize in certain token patterns.  
5. Outputs from the selected experts are combined and passed to the next Transformer layer.

This approach allows the model to:
- Learn **multiple perspectives** of symbolic equations
- Capture **global dependencies**
- Remain **modular and interpretable**

---

![src/SineKAN_MoE/sinekanmoe-architecture.png](https://github.com/Ayushmishra05/SymKAN-TP-Transformer/blob/4643dbd5a7fc87f98b7939e3735473062479dd57/src/SineKAN_MoE/sinekanmoe-architecture.png)


### Ways to Use 

1. Run the train.py file under the SineKAN-MoE directory.
2. Parameters can be updated in the config.py file

