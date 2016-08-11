
all: compliance histograms

compliance:
	python2 compliance.py

histograms: h.all.v5.0.symbols.gmt.gz \
	        cudaGSEA_seed42_hallmark.50_1048576_64.es.gz \
			broadGSEA_seed150_hallmark.50_1048576_64.edb.gz \
			broadGSEA_seed149_hallmark.50_1048576_64.edb.gz
	python2 histograms.py

h.all.v5.0.symbols.gmt.gz:
	wget abstractnonsense.de/binaries/cudaGSEA/h.all.v5.0.symbols.gmt.gz

cudaGSEA_seed42_hallmark.50_1048576_64.es.gz:
	wget abstractnonsense.de/binaries/cudaGSEA/cudaGSEA_seed42_hallmark.50_1048576_64.es.gz

broadGSEA_seed150_hallmark.50_1048576_64.edb.gz:
	wget abstractnonsense.de/binaries/cudaGSEA/broadGSEA_seed150_hallmark.50_1048576_64.edb.gz

broadGSEA_seed149_hallmark.50_1048576_64.edb.gz:
	wget abstractnonsense.de/binaries/cudaGSEA/broadGSEA_seed149_hallmark.50_1048576_64.edb.gz

clean:
	rm -f images/*
	rm -f *.gz
