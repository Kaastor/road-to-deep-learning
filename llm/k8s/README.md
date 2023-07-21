## K8s

### For quick tests

- Create GPU A100 40GB node-pool (or T4 with more mem)
- Give little more memory (~200GB)
- Create Pod: `kubectl apply -f pod.yaml`
- Get into pod: `kubectl exec -it pytorch-pod -- /bin/bash`
- Clone Llama repository `git clone https://github.com/facebookresearch/llama.git`