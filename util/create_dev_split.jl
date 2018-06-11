slc = ARGS[1] # short lang code
llc = ARGS[2] # long lang  code
dev_percent = 6

outdir = "/kuacc/users/edayanik16/safe_data/devset-generation"
trndata="/kuacc/users/edayanik16/safe_data/release-2.2-st-train-dev-data/ud-treebanks-v2.2/UD_$(llc)/$(slc)-ud-train.conllu"
all_data = Any[]
open(trndata,"r") do fin
    instance = Any[]
    while !eof(fin)
        line = readline(fin)
        if line == ""
            push!(all_data,instance)
            instance = Any[]
        else
            push!(instance,line)
        end
    end
end

ix = randperm(length(all_data))
dev_ix = ix[1:Int(floor(dev_percent*length(all_data)/100))]
trn_ix = ix[Int(floor(dev_percent*length(all_data)/100))+1:end]

trn_set = all_data[trn_ix]
dev_set = all_data[dev_ix]


newtrndata = joinpath(outdir,"UD_$(llc)/$(slc)-ud-train.conllu")
newdevdata = joinpath(outdir,"UD_$(llc)/$(slc)-ud-dev.conllu")

open(newtrndata,"w") do fout
    for i = 1:length(trn_set)
        for j =1:length(trn_set[i])
            write(fout,trn_set[i][j],"\n")
        end
        write(fout,"\n")
    end
end

open(newdevdata,"w") do fout
    for i = 1:length(dev_set)
        for j =1:length(dev_set[i])
            write(fout,dev_set[i][j],"\n")
        end
        write(fout,"\n")
    end
end
