% prepare features for subject

function PrepareFeatures4Subj(rootpath)
    subids = [1,2,3,4,5,8,9,16,17,19,20,21,22,23,24,25,30,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,63,64,65,67,68,69,70,71,72,74,75,76,77,79,80,81,82,83,84,85,86];
    totalsubs = length(subids);
    
    for i=1:totalsubs
       eeg_fea = load(strcat(rootpath, num2str(subids(i)), '/eegfea.mat'));
       per_fea = load(strcat(rootpath, num2str(subids(i)), '/perifea.mat'));
       video_fea = load(strcat(rootpath, num2str(subids(i)), '/videoFea_.mat'));
       datas = load(strcat(rootpath, num2str(subids(i)), '/datas.mat'));
       
       feas = [];
       vids = [];
       for vid=0:6
           eeg_f_ = eeg_fea.feas(eeg_fea.vids==vid,:,:);
           [r,c,p] = size(eeg_f_);
           eeg_f_ = reshape(eeg_f_, [r, c*p]);
           gsr_f_ = per_fea.feas_gsr(per_fea.vids==vid, :);
           ppg_f_ = per_fea.feas_ppg(per_fea.vids==vid, :);
           video_f = video_fea.lbps_all(video_fea.vids==vid, :);
           feas = [feas;[eeg_f_(5:end,:), gsr_f_, ppg_f_, video_f(5:end,:)]];
           
           vids = [vids;ones(r-4,1)*vid];
       end
       save(strcat(rootpath, '/features/', num2str(subids(i)),'.mat'), 'feas', 'vids');
    end
end