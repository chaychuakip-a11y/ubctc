ps ux | grep -v grep | grep 'train_dcnn'
ps ux | grep -v grep | grep 'train_dcnn' | awk '{print $2}' | xargs kill -9
