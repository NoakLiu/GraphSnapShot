# ogbn-products, ogbn-arxiv
## FBL
setting
```
sampler = NeighborSampler(
    [20, 20, 20],  # fanout for [layer-0, layer-1, layer-2]
    prefetch_node_feats=["feat"],
    prefetch_labels=["label"],
    fused=fused_sampling,
)
```
Performance
```
lstime.mean: 0.43414221542158377 s
lsmem.mean 116.24373829588015 MB
```

## FCR

arxiv
setting
```

```

```
lstime.mean: 0.5516397256529733 s
lsmem.mean: 21.298338014981272 MB
```

products
```
lstime.mean: 0.3341255570940403 s
lsmem.mean 6685.240150043178 MB
```


## FCR-SC


```
lstime.mean: 0.46770025639051804 s
lsmem.mean: 133.97237827715355 MB
```


## OTF
```

```

```
lstime.mean: 0.05882716878914051 s
lsmem.mean 6310.801840457686 MB
```

## OTF-SC
```

```

```
lstime.mean: 0.05370364535040188
lsmem.mean 3489.088622625216
```

## OTCR
```

```

## OTCR-SC
```

```

# ogbn-mag