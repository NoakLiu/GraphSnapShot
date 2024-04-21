Dense Graph GraphSnapShot Cache for SSDReS_Samplers

# method
- For sparse graphs, FBL method will be directedly deployed
- For dense graphs, SSDReS methods will be deployed

# SSDReS method
- dgl samplers
    - hete
        - FCR_hete
        - FCR_SC_hete
        - OTF((PR, FR)x(PF, FF))_hete
        - OTF((PR, FR)x(PF, FF))_SC_hete
    - homo
        - FCR
        - FCR_SC
        - OTF((PR, FR)x(PF, FF))
        - OTF((PR, FR)x(PF, FF))_SC

- dgl samplers simple
    - hete
        - FCR_hete
        - FCR_SC_hete
        - OTF_hete
        - OTF_SC_hete
    - homo
        - FCR
        - FCR_SC
        - OTF
        - OTF_SC

## figure
![model construction](./assets/dense_proc.png)

## Deployment Sequence
For homograph
    1. python div_graph_by_deg_homo.py --> dense graph, sparse graph
    2. deploy homo SSDReS samplers such as FCR, FCR-SC, OTF((PR, FR)x(PF, FF)), OTF((PR, FR)x(PF, FF))-SC on dense graph
    3. deploy FBL on sparse graph

For hetegraph
    1. python div_graph_by_deg_hete.py --> dense graph, sparse graph
    2. deploy homo SSDReS samplers such as FCR_hete, FCR-SC_hete, OTF((PR, FR)x(PF, FF))_hete, OTF((PR, FR)x(PF, FF))-SC_hete on dense graph
    3. deploy FBL on sparse graph

## figure (mem reduction-dataset, test on training)

![homo mem reduction](/results/homo/sample_efficiency_homo_arxiv.png)

![homo mem reduction](/results/homo/sample_efficiency_homo_products.png)

![hete mem reduction](/results/hete/sample_efficiency_hete_mag.png)

## analysis
The key point of GraphSnapShot is to cache the local structure instead of whole graph input for memory reduction and sampling efficiency.
