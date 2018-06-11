function stats(file)
    col4 = Dict()
    col6 = Dict()
    for line in eachline(file)
        if line == ""
            continue
        elseif (m = match(r"^(\d-*\d*)\t(.+?)\t.+?\t(.+?)\t.+?\t(.+?)\t(.+?)\t(.+?)(:.+)?\t",line)) != nothing
            id  = m.captures[1]
            contains(id,"-") && continue
            postag = m.captures[3]
            if postag == "_"
                println(line)
            end
            if haskey(col4,postag)
                col4[postag] += 1
            else
                col4[postag] = 1
            end

            midtag = m.captures[4]
            tags = split(midtag,"|")
            length(tags) == 1 && continue
            for t in tags
                if t == "_"
                    println(line)
                end
                if haskey(col6,t)
                    col6[t] += 1
                else
                    col6[t] = 1
                end
            end
        end
    end
    return col4,col6
end

function foo1(d)
    d2 = Dict()
    for k in keys(d)
        subk = split(k,"=")[1]
        if haskey(d2,subk)
            if !haskey(d2[subk],split(k,"=")[2])
                d2[subk][split(k,"=")[2]] = d[k]
            end
        else
            d2[subk] = Dict()
            d2[subk][split(k,"=")[2]] = d[k]
        end
    end
    return d2
end

function perpcalculate(d)
    vals = filter(x->x!= 0 ,collect(values(d)))
    probs = map(x->x/sum(vals),vals)
    p = exp(sum(map(x->x*(log(e,1/x)),probs)))
    return p
end

paths = split(readstring(`find /kuacc/users/edayanik16/scratch/conll18-data/release-2.2-st-train-dev-data/ud-treebanks-v2.2 -type f -name "*train.conllu"`),'\n')[1:end-1]
#paths = split(readstring(`find /ai/data/nlp/conll17/ud-treebanks-v2.0/ -type f -name "*train.conllu"`),'\n')[1:end-1]

labels = Any[]
global labels2 = Any[]
for p in paths
    c4,c6 = stats(p)
    c6v2 = foo1(c6)
    labels2 = vcat(labels2,keys(c6v2)...)
    labels  = vcat(labels,keys(c4)...)
end
labels = unique(labels)
labels2 = unique(labels2)

open("postags_combined.out","w") do fout
    firstline = string("LANGCODE","\t",join(labels,"\t"),"\n")
    write(fout,firstline)
    for p in paths
        c4,c6 = stats(p)
        langcode = split(split(p,"/")[end],"-")[1]
        ss = ""
        for x in labels
            if haskey(c4,x)
                num = c4[x]
            else
                num = 0
            end
            ss = string(ss,"\t",num)
        end
        ss = string(langcode,"\t",ss)
        write(fout,ss,"\n")
        flush(STDOUT)
    end
end

open("c6_combined.out","w") do fout
    firstline = string("LANGCODE","\t",join(labels2,"\t"),"\n")
    write(fout,firstline)
    for p in paths
        c4,c6 = stats(p)
        c6v2 = foo1(c6)
        langcode = split(split(p,"/")[end],"-")[1]
        ss = ""
        for x in labels2
            if haskey(c6v2,x)
                num = perpcalculate(c6v2[x])
            else
                num = 0
            end
            ss = string(ss,"\t",num)
        end
        ss = string(langcode,"\t",ss)
        write(fout,ss,"\n")
        flush(STDOUT)
    end
end
