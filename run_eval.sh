# 32 channels
#python val.py --model test --model_path model.pth --clip --knn 6 6 6 --nsamples 10000 5000 2500 #--save # use if saving the results --test_data_file val.list
#python val.py --model test --model_path model.pth --clip --knn 6 6 6 --nsamples 10000 5000 2500 --test_data_file val.list
#python val.py --model test --model_path model.pth --clip --knn 6 6 6 --nsamples 10000 5000 2500 --flip_input # helpful to reduce the error, as GuideNet did.#--save # use if saving the results

# 64 channels
#python val.py --model test --channels 64 --model_path model_64.pth --clip --knn 6 6 6 --nsamples 10000 5000 2500 #--save # use if saving the results
python val.py --model test --channels 64 --model_path model_64.pth --clip --knn 6 6 6 --nsamples 10000 5000 2500 --flip_input # helpful to reduce the error, as GuideNet did.#--save # use if saving the results
