%% preprocess script for raw eeg
%%
file_root={'../Personality Data/'};
for j=[1,2,3,4,5,8,9,16,17,19,20,21,22,23,24,25,30,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,63,64,65,67,68,69,70,71,72,74,75,76,77,79,80,81,82,83,84,85,86]
    file_path = [char(file_root(1)) num2str(j) '/sub' num2str(j) '/' num2str(j) '_raw.edf'];
    EEG = pop_biosig(file_path);
    
    % channel rename
    for i = 1:numel(EEG.chanlocs)
        idx =   strfind(EEG.chanlocs(i).labels,'-');
        if ~isempty(idx)
            tmp = strsplit(EEG.chanlocs(i).labels(1:idx-1),' ');
            EEG.chanlocs(i).labels = tmp{1,2};
        else
            EEG.chanlocs(i).labels = EEG.chanlocs(i).labels(5:7)
        end
    end
    
    EEG = pop_chanedit(EEG, 'lookup','C:/apps/matlab exts/eeglab2021.1/plugins/dipfit/standard_BEM/elec/standard_1005.elc'); % EEG自带地图
    EEG = pop_select( EEG,'nochannel',{'A1','A2','ger', 'X1', 'X2', 'X3'}); %去掉无用电极
    EEG = pop_reref( EEG, 9);  %重参考到CM电极
    EEG = pop_eegfiltnew(EEG, 'locutoff',1); % 高通 1Hz
    EEG = pop_eegfiltnew(EEG, 'hicutoff',50); % 低通 50Hz
    EEG = pop_eegfiltnew(EEG, 'locutoff',49,'hicutoff',51,'revfilt',1); % 凹陷 49-51Hz
    EEG = pop_rmbase( EEG, [],[]); % 去基线
    EEG = pop_saveset( EEG, 'filename','1_noICA.set', 'filepath', [char(file_root(1)) num2str(j) '/sub' num2str(j)]); 
    EEG = pop_runica(EEG, 'extended',1,'interupt','on'); % ICA
    EEG = pop_saveset( EEG, 'filename',  '1_ICA.set','filepath', [char(file_root(1)) num2str(j) '/sub' num2str(j)]); %保存数据
end