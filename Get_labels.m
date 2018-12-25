load('Anno/language_original')


% for i=1:17
%     a = find(color_==i);
%     engJ{a(1)}
% end



% for i=1:4
%     a = find(sleeve_==i);
%     engJ{a(2)}
% end

% 
% for j=1:length(a)
%     
%     engJ{a(j)}
%     
%     
% end
% 
% All_miss = engJ{a}
% sleeve_([a(40,)]) = 2
% sleeve_([a(40,)]) = 1

% sleeve_([a(48)]) = 3

% for i=0:1
%     a = find(gender_==i);
%     engJ{a(1)}
% end


% for i=1:19
%     a = find(cate_new==i);
%     engJ{a(1:3)}
% end

%%

% % Color: Red, Blue, White, Pink, Orange, Black, Brown, Green, Purple,
% % Yellow, Khaki, Gray, Beige, Blonde, Silver, Olives, Multicolor

% % Sleeve: Long, Short, Sleeveless, Others

% % Cat: Blazer, Blouse, Bomber, Cardigan, Henley, Hoodie, Jacket, Jersey,
% % Parka, Poncho, Sweater, Tank, Tee, Top, Coat, Dress, Jumpsuit, Kimono,
% % Romper

% % Gender: woman, man

fid = fopen(['Labels_Synth.txt'],'w');
imagefiles = dir('/home/emirak/Desktop/Clothing_Datasets/2016_Deep_fashion/1_Category_Attribute_Prediction/Images_Synth/*.jpg');

for j=1:length(color_)
    
    file_full = [imagefiles(j).name ' ' int2str(color_(j)-1)  ' ' int2str(sleeve_(j)-1) ' ' int2str(cate_new(j)-1) ' ' int2str(gender_(j)) '\n' ];
    fprintf(fid,file_full);
    
end