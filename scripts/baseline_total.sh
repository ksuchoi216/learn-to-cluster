
cut_name=_cut
sh=./scripts/baseline/$+.sh

for method in fast_hac #kmeans dbscan aro fast_hac
do
    echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
    echo "$@@@@@@@@@@@@@ EXECUTION[$method] @@@@@@@@@@@@@@@"

    path=./scripts/baseline/$method\_cut.sh
    sh $path

    echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

done