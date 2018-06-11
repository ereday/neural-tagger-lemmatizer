lcase(x::Char) = x=='I' ? 'ı':lowercase(x);
lcase(s::AbstractString) = map(lcase,s)

rootpath="/ai/data/nlp/conll17/ud-treebanks-v2.0"
splits = ["train","dev","test"]
langs_s = ["sv","zh","sl","el","he","pl"]
langs_l = ["Swedish","Chinese","Slovenian","Greek","Hebrew","Polish"]#["Dutch"]
#langs_s = ["ar","bg","cs","en","et","fr","hu","ro","tr","hr","it","da","ca","fi","es","pt","sk","sl","ru","hi"]
#langs_l = ["Arabic","Bulgarian","Czech","English","Estonian","French","Hungarian","Romanian","Turkish","Croatian","Italian","Danish","Catalan","Finnish","Spanish","Portuguese","Slovak","Slovenian","Russian","Hindi"]

for i=1:length(langs_s)
    token_nums = Any[]
    tags = Any[]
    feats = Any[]
    for (ix,s) in enumerate(splits)
        if ix != 3
            fname = "$(rootpath)/UD_$(langs_l[i])/$(langs_s[i])-ud-$(s).conllu"
        else
            fname = "/ai/data/nlp/conll17/ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/$(langs_s[i]).conllu"
        end
        token_num = 0
        open(fname,"r") do fin
            while ! eof(fin)
                line = readline(fin)
                if line == "" || startswith(line,"#")
                    continue
                end
                if  (m=match(r"^(\d-*\d*)\t(.+?)\t(.+?)\t(.+?)\t(.+?)\t(.+?)\t(.+?)\t(.+?)(:.+)?\t",line)) != nothing
                    token_num += 1
                    id  = m.captures[1]
                    contains(id,"-") && continue
                    word   = lcase(m.captures[2])
                    lemma  = lcase(m.captures[3])
                    postag = m.captures[4]
                    morphemes = m.captures[6]
                    ms = split(morphemes,"|")
                    if morphemes != "_"
                        map(q->push!(feats,q),ms)
                    end
                    if morphemes != "_"
                        label = unshift!(ms,postag)
                    else
                        label = [postag]
                    end
                    push!(tags,join(label,"|"))
                end
            end
            push!(token_nums,token_num)
        end
    end
    unique_tags_num = length(unique(tags))
    unique_feats_num = length(unique(feats))
    # Number of tokens in training,dev,test
    # Number of  Unique tags
    # Number of unique features
    println("$(langs_s[i])\t$(token_nums[1])\t$(token_nums[2])\t$(token_nums[3])\t$(unique_tags_num)\t$(unique_feats_num)")
end
