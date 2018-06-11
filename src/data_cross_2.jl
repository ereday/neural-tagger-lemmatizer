LOWLANG="lowlang"
HIGHLANG="highlang"
lcase(x::Char) = x=='I' ? 'ı':lowercase(x);
lcase(s::AbstractString) = map(lcase,s)
ucase(x::Char) = x=='i' ? 'İ':uppercase(x);
ucase(s::AbstractString) = map(ucase,s)
function convert_data_format(df)
    converted_data = Any[]
    for f in df
        open(f,"r") do fin
            sentence = Any[]
            sentences = Any[]
            while !eof(fin)
                line = readline(fin)
                if line == "" # sentence is over new sentence is about to start
                    push!(sentences,sentence)
                    sentence = Any[]
                elseif (m=match(r"^(\d-*\d*)\t(.+?)\t(.+?)\t(.+?)\t(.+?)\t(.+?)\t(.+?)\t(.+?)(:.+)?\t",line)) != nothing
                    id  = m.captures[1]
                    contains(id,"-") && continue
                    word   = lcase(m.captures[2])
                    lemma  = lcase(m.captures[3])
                    postag = m.captures[4]
                    morphemes = m.captures[6]
                    ms = split(morphemes,"|")
                    if morphemes != "_"
                        label = unshift!(ms,postag)
                    else
                        label = [postag]
                    end
                    push!(sentence,(word,lemma,label))
                else
                    # raw sentence starting with # symbol
                end
            end
            push!(converted_data,sentences)
        end
    end
    return converted_data
end


function create_vocabulary(cdf_low,cdf_high)
    c2i = Dict()
    o2i = Dict()
    global UNKTAG = "UNK"
    global MASK   = "MASK"
    global EOW    = "EOW"
    global BOW    = "BOW"

    for cdf in [cdf_low,cdf_high...]
        for (ix,cf) in enumerate(cdf)
            for sentence in cf
                for (w,l,t) in sentence
                    map(x->get!(c2i,x,length(c2i)+1),collect(w))
                    map(x->get!(o2i,x,length(o2i)+1),collect(w))
                end
            end
        end
        for (ix,cf) in enumerate(cdf)
            for sentence in cf
                for (w,l,t) in sentence
                    map(x->get!(o2i,x,length(o2i)+1),t)
                end
            end
        end
    end
    get!(o2i,UNKTAG,length(o2i)+1)
    get!(o2i,MASK,length(o2i)+1)
    get!(o2i,BOW,length(o2i)+1)
    get!(o2i,EOW,length(o2i)+1)
    # Add language symbols
    get!(c2i,LOWLANG,length(c2i)+1)
    counter=1
    for i=1:length(cdf_high)
        get!(c2i,string(HIGHLANG,counter),length(c2i)+1)
        counter += 1
    end
    return c2i,o2i
end

function encode_data(cdf_low,cdf_high,c2i,o2i)
    edf_low = _encode_data(cdf_low,c2i,o2i,LOWLANG)
    counter = 1
    edf_high = Any[]
    for i=1:length(cdf_high)
        edf_high_x = _encode_data(cdf_high[i],c2i,o2i,string(HIGHLANG,counter))
        push!(edf_high,edf_high_x)
        counter += 1
    end
    return edf_low,edf_high
end

function _encode_data(cdf,c2i,o2i,langcode)
    edf = Any[]
    for cf in cdf
        esentences = Any[]
        for sentence in cf
            esentence = Any[]
            for (w,l,t) in sentence
                eword  = map(x->c2i[x],collect(w))
                push!(eword,c2i[langcode])
                unshift!(eword,c2i[langcode])
                elemma = map(x->get(o2i,x,o2i[UNKTAG]),collect(l))
                etag   = map(x->o2i[x],t)
                push!(esentence,(eword,elemma,etag))
            end
            push!(esentences,esentence)
        end
        push!(edf,esentences)
    end
    return edf
end

function prepare_data(opts)
    df_low  = opts[:datafiles_low]
    df_high = opts[:datafiles_high]
    cdf_low = convert_data_format(df_low)
    cdf_high = [convert_data_format(df_high[i:i+3-1]) for i=1:3:length(df_high)]
    #cdf_high = convert_data_format(df_high)
    c2i,o2i = create_vocabulary(cdf_low,cdf_high)
    edf_low,edf_high = encode_data(cdf_low,cdf_high,c2i,o2i)
    return edf_low,edf_high,c2i,o2i
end
