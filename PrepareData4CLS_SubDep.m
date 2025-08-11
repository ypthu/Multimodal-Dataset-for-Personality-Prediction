%% prepare data for subject dependent
function PrepareData4CLS_SubDep(rootpath, do_norm)
    if nargin < 2
       do_norm = true; 
    end
    
    label_data = readmatrix(strcat(rootpath, '/Original/BFI44_Labels.csv'), 'NumHeaderLines', 1);
    
    subids = label_data(:,1);
    labels = label_data(:,2:end)';
    
    

    
    totalsubs = length(subids);
    
    for vid=0:6
        Feas = [];
        Labels = [];
        %testFea = [];
        %testLabel = [];
        
        for i=1:totalsubs
           alldata = load(strcat(rootpath, '/Features/', num2str(subids(i)),'.mat'));

           feas = alldata.feas;
           vids = alldata.vids;
           %dislabel = alldata.dis_label;
           %dataLabels = labels(vids'+1);


           feas4v = feas(vids==vid,:);
           [r, c] = size(feas4v);
           if do_norm
              min_ = min(feas4v);
              max_ = max(feas4v);
              feas4v = (feas4v-repmat(min_, r,1))./(repmat(max_-min_+0.0000001, r, 1));
           end

           [r,c] = size(feas4v);

           Feas = [Feas;feas4v];
           Labels = [Labels, ones(5, r).* repmat(labels(:,i), 1, r)];
           %testFea = [testFea;feas4v((ceil(r*4/5)+1):end,:)];
           %testLabel = [testLabel, ones(5, r-ceil(r*4/5)).* repmat(labels(:,vid+1), 1, r-ceil(r*4/5))];

        end
        
        Feas = real(Feas);
        
        if do_norm
            save(['./AllData/' num2str(vid) '_norm.mat'], 'Feas', 'Labels');
        else
            save(['./AllData/' num2str(vid) '.mat'], 'Feas', 'Labels');
        end
        
    end
end