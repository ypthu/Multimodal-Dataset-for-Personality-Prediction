% format preprocessed data

rootpath = '../Personality Data/'
TOTAL_TRIS=18;

invalid = []
for j=[1,2,3,4,5,8,9,16,17,19,20,21,22,23,24,25,30,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,63,64,65,67,68,69,70,71,72,74,75,76,77,79,80,81,82,83,84,85,86]
    display(strcat('Subject: ', num2str(j)));
%    %% label distribution
%    panas_f = dir([rootpath num2str(j) '/' num2str(j) '/*_panas.csv']);
%    panas = csvread([rootpath num2str(j) '/' num2str(j) '/' panas_f.name]);
%    [r, inds]=sort(panas(:,1));
%    panas_=panas(inds, 2:end);
%    if j==26
%       panas_ = panas_ - 10; 
%    end
%    s = sum(panas_, 2);
%    s = repmat(s, [1, 10]);
%    dis_label = panas_./s;
   
   %% EEG
   file_name = '1_ICA_remcop.set'
   file_name_o = '1_ICA.set'
   EEG = pop_loadset('filename',file_name, 'filepath',[rootpath num2str(j) '/sub' num2str(j) '/']) 
   EEG_o = pop_biosig([rootpath num2str(j) '/sub' num2str(j) '/' num2str(j) '_raw.edf']); 
   triggers =EEG_o.data(25,:);
   tri_inds = find(triggers > 0);
   tris = triggers(tri_inds);
   bg = find(tris==60);
   ed = size(tris,2);
   tri_inds_t= [];
   tris_t = [];
   for cur = bg:ed-1
       if (tris(cur+1)-tris(cur)) == 90
           tri_inds_t = [tri_inds_t, tri_inds(cur),tri_inds(cur+1)];
           tris_t = [tris_t, tris(cur), tris(cur+1)];
       end
   end
   tri_inds = tri_inds_t;%tri_inds(bg:end);
   tris = tris_t;%tris(bg:end);
   
   eeg_datas = [];
   if size(tri_inds, 2) == TOTAL_TRIS
       assert(tris(3)==50)
       for i=3:2:TOTAL_TRIS
          assert((tris(i+1)-tris(i)) == 90)
          eeg_data = EEG.data(:, tri_inds(i):tri_inds(i+1)-1);
          vids = ones(1, size(eeg_data, 2))*(tris(i)-10);
          eeg_datas = [eeg_datas, [eeg_data;vids]];
       end
   end
   if ~exist([rootpath num2str(j) '/sub' num2str(j) '/splitraw'], 'dir')
      mkdir([rootpath num2str(j) '/sub' num2str(j) '/splitraw']) 
   end
%    save([rootpath num2str(j) '/sub' num2str(j) '/splitraw/eeg_data.mat'], 'eeg_datas');
   
   %% PPG and GSR
   gsr_ = csvread([rootpath num2str(j) '/' num2str(j) '/raw_gsr.csv']);
   gsr_data = gsr_(:,2)';
   triggers = gsr_(:,3)';
   tri_inds = find(triggers > 0);
   tris = triggers(tri_inds);
   bg = find(tris==60);
   ed = size(tris,2);
   tri_inds_t= [];
   tris_t = [];
   for cur = bg:ed-1
       if (tris(cur+1)-tris(cur)) == 90
           tri_inds_t = [tri_inds_t, tri_inds(cur),tri_inds(cur+1)];
           tris_t = [tris_t, tris(cur), tris(cur+1)];
       end
   end
   tri_inds = tri_inds_t;
   tris = tris_t;
   assert(tris(3)==50)
   gsr_datas = []
   if size(tri_inds, 2)==TOTAL_TRIS
       for i=3:2:TOTAL_TRIS
          assert((tris(i+1)-tris(i)) == 90)
          data_ = gsr_data(:, tri_inds(i):tri_inds(i+1)-1);
          vids = ones(1, size(data_, 2))*(tris(i)-10);
          gsr_datas = [gsr_datas, [data_;vids]];
       end
   end
   
   %% gsr_fea
   gsr_fea = csvread([rootpath num2str(j) '/' num2str(j) '/fea_gsr.csv']);
   gsr_fea_data = gsr_fea(:,2)';
   triggers = gsr_fea(:,3)';
   tri_inds = find(triggers > 0);
   tris = triggers(tri_inds);
   bg = find(tris==60);
   ed = size(tris,2);
   tri_inds_t= [];
   tris_t = [];
   for cur = bg:ed-1
       if (tris(cur+1)-tris(cur)) == 90
           tri_inds_t = [tri_inds_t, tri_inds(cur),tri_inds(cur+1)];
           tris_t = [tris_t, tris(cur), tris(cur+1)];
       end
   end
   tri_inds = tri_inds_t;
   tris = tris_t;
   assert(tris(3)==50)
   gsr_fea_datas = []
   if size(tri_inds, 2)==TOTAL_TRIS
       for i=3:2:TOTAL_TRIS
          assert((tris(i+1)-tris(i)) == 90)
          data_fea = gsr_fea_data(:, tri_inds(i):tri_inds(i+1)-1);
          vids = ones(1, size(data_fea, 2))*(tris(i)-10);
          gsr_fea_datas = [gsr_fea_datas, [data_fea;vids]];
       end
   end
   
   ppg_ = csvread([rootpath num2str(j) '/' num2str(j) '/raw_ppg.csv']);
   ppg_data = ppg_(:,2)';
   triggers = ppg_(:,3)';
   tri_inds = find(triggers > 0);
   tris = triggers(tri_inds);
   bg = find(tris==60);
   ed = size(tris,2);
   tri_inds_t= [];
   tris_t = [];
   for cur = bg:ed-1
       if (tris(cur+1)-tris(cur)) == 90
           tri_inds_t = [tri_inds_t, tri_inds(cur),tri_inds(cur+1)];
           tris_t = [tris_t, tris(cur), tris(cur+1)];
       end
   end
   tri_inds = tri_inds_t;
   tris = tris_t;
   assert(tris(3)==50)
   ppg_datas = []
   if size(tri_inds, 2) ==TOTAL_TRIS
       for i=3:2:TOTAL_TRIS
          assert((tris(i+1)-tris(i)) == 90)
          data_ = ppg_data(:, tri_inds(i):tri_inds(i+1)-1);
          vids = ones(1, size(data_, 2))*(tris(i)-10);
          ppg_datas = [ppg_datas, [data_;vids]];
       end
   end
   
   %% ppg_fea
   ppg_fea = csvread([rootpath num2str(j) '/' num2str(j) '/fea_ppg.csv']);
   ppg_fea_data = ppg_fea(:,2)';
   triggers = ppg_fea(:,3)';
   tri_inds = find(triggers > 0);
   tris = triggers(tri_inds);
   bg = find(tris==60);
   ed = size(tris,2);
   tri_inds_t= [];
   tris_t = [];
   for cur = bg:ed-1
       if (tris(cur+1)-tris(cur)) == 90
           tri_inds_t = [tri_inds_t, tri_inds(cur),tri_inds(cur+1)];
           tris_t = [tris_t, tris(cur), tris(cur+1)];
       end
   end
   tri_inds = tri_inds_t;
   tris = tris_t;
   assert(tris(3)==50)
   ppg_fea_datas = []
   if size(tri_inds, 2) ==TOTAL_TRIS
       for i=3:2:TOTAL_TRIS
          assert((tris(i+1)-tris(i)) == 90)
          data_fea = ppg_fea_data(:, tri_inds(i):tri_inds(i+1)-1);
          vids = ones(1, size(data_fea, 2))*(tris(i)-10);
          ppg_fea_datas = [ppg_fea_datas, [data_fea;vids]];
       end
   end
   
   if size(gsr_datas, 2)==0 || size(ppg_datas, 2)==0
       invalid = [invalid, j];
   end
   
   
   save([rootpath num2str(j) '/sub' num2str(j) '/splitraw/eeg_data.mat'], 'eeg_datas', 'gsr_datas', 'gsr_fea_datas', 'ppg_datas', 'ppg_fea_datas');
end

