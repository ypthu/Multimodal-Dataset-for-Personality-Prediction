clear;clc;eeglab
%% 

%% 
file_root={'../Personality Data/'};
for j=33:87
    [file_root num2str(j) '/sub' num2str(j) '/1_ICA.set']
    if exist([char(file_root(1)) num2str(j) '/sub' num2str(j) '/1_ICA.set'], 'file')~=2
        continue;
    end
    file_name = '1_ICA.set';
    EEG = pop_loadset('filename',file_name, 'filepath',[char(file_root(1)) num2str(j) '/sub' num2str(j) '/'])
    %EEG = pop_runica(EEG, 'extended',1,'pca',50,'interupt','on');%runICA
	%EEG = pop_runica(EEG, 'extended',1,'interupt','on');%runICA
	EEG = iclabel(EEG);
	noiselabel = round(EEG.etc.ic_classification.ICLabel.classifications(:,:)*100)
	
	noisethreshold = [0 0; 0.9 1;0.9 1;0 0;0 0;0 0;0 0];
	EEG = pop_icflag(EEG, noisethreshold);
	EEG = pop_subcomp(EEG, []);
    
	EEG = pop_saveset( EEG, 'filename','1_ICA_remcop.set','filepath', [char(file_root(1)) num2str(j) '/sub' num2str(j) '/']); %保存数据
end
