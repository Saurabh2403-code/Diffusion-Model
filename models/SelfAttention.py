class SelfAttentionBlock(nn.Module):
    def __init__(self,no_of_color_channels_in_input,no_of_dimension_in_embedding_space):
        super().__init__()
        self.dimension=no_of_dimension_in_embedding_space
        self.color_channels=no_of_color_channels_in_input
        self.Query=nn.Linear(no_of_color_channels_in_input,no_of_dimension_in_embedding_space)
        self.Key=nn.Linear(no_of_color_channels_in_input,no_of_dimension_in_embedding_space)
        self.Value=nn.Linear(no_of_color_channels_in_input,no_of_dimension_in_embedding_space)
        self.norm=nn.GroupNorm(8,no_of_color_channels_in_input)
        self.out_proj = nn.Linear(no_of_dimension_in_embedding_space,no_of_color_channels_in_input)
    def forward(self,x):
        x_0=x
        B,C,H,W=x.shape
        x=self.norm(x)
        x=x.reshape(B,C,H*W).transpose(1,2)
        query=self.Query(x)
        key=self.Key(x)
        values=self.Value(x)
        similarity_score=(query@key.transpose(-2,-1))/(torch.sqrt(torch.tensor(self.dimension,dtype=torch.float32,device=x.device)))
        weights_for_values=F.softmax(similarity_score,dim=-1)

        weighted_values=weights_for_values@values
        
        weighted_values = self.out_proj(weighted_values)

        weighted_values=weighted_values.transpose(1,2).reshape_as(x_0)

        final=x_0+weighted_values
        return final
