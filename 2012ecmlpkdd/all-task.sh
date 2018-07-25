### task list
### included from gxp-all-task.sh or test-all-task.sh

##############################

script=learn-cv-lr.sh

ntry=1
for reg in 1; do
for ltype in 4; do
for itype in 3; do
for eta in 5.0; do

# prejudice remover
method=PR${ltype}
lscript=train_pr.py
tscript=predict_lr.py
go "reg=${reg} eta=${eta} ltype=${ltype} itype=${itype} try=${ntry}"

done
done
done
done

##############################
