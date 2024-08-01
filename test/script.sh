# Test all methods for fp32 with weight_decay
python test/test_method.py --batch 16 --n 25600 --method causal_dot_product --type float32 --gpu 5 --is_weight_decay --onlyMethod
# python test/test_method.py --batch 2 --n 1024 --method lightningAttention --type float32 --gpu 5 --is_weight_decay 
# python test/test_method.py --batch 2 --n 1024 --method blockbased --type float32 --gpu 5 --is_weight_decay 
# python test/test_method.py --batch 2 --n 1024 --method rowbased --type float32 --gpu 5 --is_weight_decay 
# python test/test_method.py --batch 2 --n 1024 --method BCMV_vanilla --type float32 --gpu 5 --is_weight_decay 
# python test/test_method.py --batch 2 --n 1024 --method FleetAttention --type float32 --gpu 5 --is_weight_decay 
# python test/test_method.py --batch 2 --n 1024 --method recursion --type float32 --gpu 5 --is_weight_decay 


